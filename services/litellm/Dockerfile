# LiteLLM Proxy for AWS Bedrock
# Unified LLM + Embedding gateway

FROM ghcr.io/berriai/litellm:main-latest

# Copy config file
COPY config.yaml /app/config.yaml

# Set config path
ENV LITELLM_CONFIG_PATH=/app/config.yaml

EXPOSE 4000

# Start proxy on port 4000
CMD ["--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "4000", "--num_workers", "1"]
