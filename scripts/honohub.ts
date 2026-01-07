#!/usr/bin/env bun
/**
 * HonoHub - HTTP reverse proxy for LightRAG Docker services
 * Proxies requests from localhost to Docker gateway with proper header rewriting
 *
 * This is useful when running in environments like:
 * - Kubernetes pods
 * - code-server / VS Code Remote
 * - Corporate proxy environments
 * - Docker Desktop with non-localhost bindings
 *
 * Usage:
 *   bun run scripts/honohub.ts
 *   HONOHUB_FORCE=true bun run scripts/honohub.ts  # Force proxy even if localhost works
 */

import { Hono } from "hono";
import { $ } from "bun";
import { existsSync, readFileSync } from "fs";
import { join, dirname } from "path";

// Load .env from parent directory (LightRAG root)
const scriptDir = dirname(import.meta.path);
const envPath = join(scriptDir, "..", ".env");
if (existsSync(envPath)) {
  const envContent = readFileSync(envPath, "utf-8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (trimmed && !trimmed.startsWith("#")) {
      const [key, ...valueParts] = trimmed.split("=");
      const value = valueParts.join("=");
      if (key && value !== undefined && !process.env[key]) {
        process.env[key] = value;
      }
    }
  }
}

// LightRAG service port mappings
// For services with targetPort, HonoHub proxies port -> targetPort on specified host
// For services without targetPort, HonoHub proxies port -> port on Docker gateway
const PORT_MAPPINGS = [
  { port: 9621, targetPort: 9622, targetHost: "127.0.0.1", name: "LightRAG API + WebUI" }, // Runs locally on 9622
  { port: 4000, name: "LiteLLM Proxy" },
  { port: 5173, name: "Vite Dev Server" },
  { port: 9100, name: "RustFS S3 API" },
  { port: 9101, name: "RustFS Web Console" },
  { port: 5432, name: "PostgreSQL", skip: true }, // TCP, not HTTP - skip by default
];

// Filter to only HTTP-compatible services
const HTTP_PORTS = PORT_MAPPINGS.filter((m) => !m.skip);
const PORTS = HTTP_PORTS.map((m) => m.port);

async function getServiceHost(): Promise<string> {
  // Strategy: Try to detect the gateway IP where services are bound

  // 1. Try lightrag-specific Docker network first
  const networkNames = [
    "lightrag-stack_lightrag-network",
    "lightrag_default",
    "lightrag-network",
  ];

  for (const networkName of networkNames) {
    try {
      const result = await $`docker network inspect ${networkName}`.text();
      const networkInfo = JSON.parse(result);
      const gateway = networkInfo[0]?.IPAM?.Config?.[0]?.Gateway;

      if (gateway) {
        console.log(`✓ Detected ${networkName} gateway: ${gateway}`);
        // Test if we can reach the main LightRAG port on this gateway
        try {
          const response = await fetch(`http://${gateway}:9621/health`, {
            signal: AbortSignal.timeout(2000),
          });
          console.log(`✓ LightRAG API accessible at ${gateway}:9621`);
          return gateway;
        } catch {
          console.log(
            `  ℹ️  Gateway ${gateway} detected but services not accessible`
          );
          console.log(
            `  ℹ️  This usually means DOCKER_GATEWAY_IP in .env is set to 127.0.0.1`
          );
        }
      }
    } catch {
      // Network doesn't exist, try next
    }
  }

  // 2. Try to get any Docker bridge network gateway
  try {
    const result = await $`docker network inspect bridge`.text();
    const networkInfo = JSON.parse(result);
    const gateway = networkInfo[0]?.IPAM?.Config?.[0]?.Gateway;

    if (gateway) {
      console.log(`  Trying Docker bridge gateway: ${gateway}`);
      try {
        const response = await fetch(`http://${gateway}:9621/health`, {
          signal: AbortSignal.timeout(2000),
        });
        console.log(`✓ LightRAG API accessible at ${gateway}:9621`);
        return gateway;
      } catch {
        // Not accessible
      }
    }
  } catch {
    // Bridge network not available
  }

  // 3. Fallback: try common Docker gateway IPs
  console.log("\n  Trying fallback gateway IPs...");
  const fallbacks = [
    "172.19.0.1",
    "172.18.0.1",
    "172.17.0.1",
    "host.docker.internal",
  ];
  for (const ip of fallbacks) {
    try {
      const response = await fetch(`http://${ip}:9621/health`, {
        signal: AbortSignal.timeout(1500),
      });
      console.log(`✓ Using fallback gateway: ${ip}`);
      return ip;
    } catch {
      console.log(`  ✗ ${ip} not accessible`);
    }
  }

  console.warn("\n⚠ No accessible gateway found. Services may not be reachable.");
  console.warn("⚠ Try running: docker-compose up -d");
  return "127.0.0.1"; // Last resort
}

function createProxyApp(targetHost: string, targetPort: number) {
  const app = new Hono();

  // Proxy all requests
  app.all("*", async (c) => {
    const incomingUrl = new URL(c.req.raw.url);
    const targetUrl = `http://${targetHost}:${targetPort}${incomingUrl.pathname}${incomingUrl.search}`;

    try {
      // Check if this is a WebSocket upgrade request
      const upgrade = c.req.header("upgrade");
      if (upgrade?.toLowerCase() === "websocket") {
        console.log(`WebSocket upgrade request to ${targetUrl}`);
        return c.text("WebSocket proxying not yet implemented", 501);
      }

      // Build headers with proper proxy forwarding
      const headers = new Headers();
      c.req.raw.headers.forEach((value, key) => {
        if (key.toLowerCase() !== "host") {
          headers.set(key, value);
        }
      });
      headers.set("host", `${targetHost}:${targetPort}`);

      // Pass X-Forwarded headers so FastAPI knows about the proxy
      // These help with correct redirect URL generation
      if (!headers.has("X-Forwarded-Host")) {
        headers.set("X-Forwarded-Host", c.req.header("host") || "localhost");
      }
      if (!headers.has("X-Forwarded-Proto")) {
        headers.set("X-Forwarded-Proto", incomingUrl.protocol.replace(":", ""));
      }
      if (!headers.has("X-Forwarded-For")) {
        headers.set("X-Forwarded-For", "127.0.0.1");
      }

      // Forward request
      const response = await fetch(targetUrl, {
        method: c.req.method,
        headers,
        body:
          c.req.method !== "GET" && c.req.method !== "HEAD"
            ? await c.req.raw.arrayBuffer()
            : undefined,
      });

      // Build response headers
      const responseHeaders = new Headers();
      response.headers.forEach((value, key) => {
        const lowerKey = key.toLowerCase();

        // Skip encoding headers - fetch() already decoded the response
        if (lowerKey === "content-encoding" || lowerKey === "content-length") {
          return;
        }

        // Rewrite Location header for redirects
        if (lowerKey === "location") {
          if (value.startsWith("/")) {
            responseHeaders.set(key, value);
          } else if (value.includes(targetHost)) {
            const rewritten = value.replace(
              new RegExp(`http://${targetHost}:${targetPort}`, "g"),
              `http://localhost:${targetPort}`
            );
            responseHeaders.set(key, rewritten);
          } else {
            responseHeaders.set(key, value);
          }
        } else {
          responseHeaders.set(key, value);
        }
      });

      // Check if response is HTML and needs path rewriting
      const contentType = response.headers.get("content-type");
      if (contentType?.includes("text/html")) {
        let html = await response.text();

        // Inject base tag if not already present
        if (!html.includes("<base")) {
          html = html.replace(/<head>/i, '<head>\n    <base href="./">');
        }

        // Inject fetch/XMLHttpRequest interceptor for SPA API calls
        const interceptorScript = `
    <script>
      // Intercept fetch to rewrite absolute URLs to relative
      (function() {
        const originalFetch = window.fetch;
        window.fetch = function(url, options) {
          if (typeof url === 'string' && url.startsWith('/') && !url.startsWith('//')) {
            // Don't rewrite API calls - they need to go through the proxy as-is
            if (!url.startsWith('/api/')) {
              url = '.' + url;
            }
          }
          return originalFetch(url, options);
        };

        // Intercept XMLHttpRequest for older apps
        const originalOpen = XMLHttpRequest.prototype.open;
        XMLHttpRequest.prototype.open = function(method, url, ...rest) {
          if (typeof url === 'string' && url.startsWith('/') && !url.startsWith('//')) {
            if (!url.startsWith('/api/')) {
              url = '.' + url;
            }
          }
          return originalOpen.call(this, method, url, ...rest);
        };

        // Intercept iframe src attribute setting
        const iframeDescriptor = Object.getOwnPropertyDescriptor(HTMLIFrameElement.prototype, 'src');
        if (iframeDescriptor && iframeDescriptor.set) {
          Object.defineProperty(HTMLIFrameElement.prototype, 'src', {
            set: function(value) {
              if (typeof value === 'string' && value.startsWith('/') && !value.startsWith('//')) {
                value = '.' + value;
              }
              iframeDescriptor.set.call(this, value);
            },
            get: iframeDescriptor.get
          });
        }
      })();
    </script>`;

        html = html.replace(/<head>/i, "<head>" + interceptorScript);

        // Rewrite absolute paths to relative paths
        html = html.replace(/\s(src|href)="\/(?!\/)/gi, ' $1="./');

        return new Response(html, {
          status: response.status,
          statusText: response.statusText,
          headers: responseHeaders,
        });
      }

      // Return proxied response for non-HTML
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });
    } catch (error) {
      console.error(`Error proxying to ${targetUrl}:`, error);
      return c.text(`Proxy error: ${error}`, 502);
    }
  });

  return app;
}

async function checkIfServicesAccessibleOnLocalhost(): Promise<boolean> {
  // Check LightRAG API or LiteLLM - either indicates services are on localhost
  const endpoints = [
    "http://127.0.0.1:9621/health",
    "http://127.0.0.1:4000/health",
  ];

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, {
        signal: AbortSignal.timeout(2000),
      });
      return true;
    } catch {
      // Try next endpoint
    }
  }
  return false;
}

async function main() {
  console.log("🚀 Starting HonoHub for LightRAG...\n");
  console.log("   LightRAG HTTP reverse proxy for Docker/K8s environments\n");

  const forceProxy =
    process.env.HONOHUB_FORCE === "true" || process.env.HONOHUB_FORCE === "1";

  // Auto-enable proxy when ROOT_PATH is set (indicates reverse proxy environment)
  const hasRootPath = !!process.env.ROOT_PATH;

  const servicesOnLocalhost = await checkIfServicesAccessibleOnLocalhost();

  if (forceProxy || hasRootPath) {
    if (hasRootPath) {
      console.log(`ℹ️  ROOT_PATH detected (${process.env.ROOT_PATH})`);
      console.log("ℹ️  Running proxy for path/redirect rewriting\n");
    } else {
      console.log(
        "ℹ️  HONOHUB_FORCE enabled - running proxy for path/redirect rewriting"
      );
    }
  } else if (servicesOnLocalhost) {
    console.log("ℹ️  LightRAG services are already accessible on localhost");
    console.log("ℹ️  HonoHub proxy is not needed in this environment");
    console.log(
      "ℹ️  Set HONOHUB_FORCE=true if you need path rewriting for proxies\n"
    );
    console.log("✓ Services available:");
    for (const mapping of HTTP_PORTS) {
      console.log(`  - ${mapping.name}: http://localhost:${mapping.port}`);
    }
    console.log("\n👋 Exiting...\n");
    process.exit(0);
  }

  // Auto-detect the gateway IP
  const serviceHost = await getServiceHost();
  console.log("");

  // Start proxy servers
  const servers: { port: number; server: ReturnType<typeof Bun.serve> }[] = [];

  for (const mapping of HTTP_PORTS) {
    try {
      // Use targetHost/targetPort if specified, otherwise use Docker gateway
      const proxyHost = (mapping as any).targetHost || serviceHost;
      const proxyPort = (mapping as any).targetPort || mapping.port;

      const app = createProxyApp(proxyHost, proxyPort);

      const server = Bun.serve({
        port: mapping.port,
        hostname: "127.0.0.1",
        fetch: app.fetch,
      });

      servers.push({ port: mapping.port, server });
      console.log(
        `→ Proxying localhost:${mapping.port} -> ${proxyHost}:${proxyPort} (${mapping.name})`
      );
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      // Port might already be in use or service might not be running
      if (message.includes("EADDRINUSE")) {
        console.log(
          `  ⚠ Port ${mapping.port} already in use (${mapping.name}) - skipping`
        );
      } else {
        console.log(
          `  ✗ Failed to proxy port ${mapping.port} (${mapping.name}): ${message}`
        );
      }
    }
  }

  if (servers.length === 0) {
    console.error("\n❌ No ports could be proxied. Exiting.");
    process.exit(1);
  }

  console.log(
    `\n✓ Proxying ${servers.length}/${HTTP_PORTS.length} services.\n`
  );
  console.log("📍 Access points:");
  for (const { port } of servers) {
    const mapping = HTTP_PORTS.find((m) => m.port === port);
    console.log(`   http://localhost:${port}  (${mapping?.name})`);
  }
  console.log("\n   Press Ctrl+C to stop.\n");

  return servers;
}

// Signal handlers
let shouldRestart = true;
const RESTART_DELAY_MS = 2000;

process.on("SIGINT", () => {
  console.log("\n\n👋 Stopping HonoHub...");
  shouldRestart = false;
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\n\n👋 Stopping HonoHub...");
  shouldRestart = false;
  process.exit(0);
});

async function runWithAutoRestart() {
  let servers: { port: number; server: ReturnType<typeof Bun.serve> }[] = [];

  while (shouldRestart) {
    try {
      servers = await main();
      // Wait indefinitely
      await new Promise(() => {});
    } catch (error) {
      console.error("\n❌ HonoHub crashed:", error);

      try {
        servers.forEach(({ server }) => server?.stop());
      } catch (cleanupError) {
        console.error("Error during cleanup:", cleanupError);
      }

      if (shouldRestart) {
        console.log(`🔄 Restarting in ${RESTART_DELAY_MS / 1000} seconds...\n`);
        await new Promise((resolve) => setTimeout(resolve, RESTART_DELAY_MS));
      } else {
        break;
      }
    }
  }
}

runWithAutoRestart().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
