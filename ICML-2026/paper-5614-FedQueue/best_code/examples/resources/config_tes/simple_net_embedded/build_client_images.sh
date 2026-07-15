#!/bin/bash
# Build client-specific Docker images with embedded data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building client-specific Docker images with embedded data...${NC}"

# Check if data exists
if [ ! -d "/tmp/tes-data" ]; then
    echo -e "${RED}Error: Data directory /tmp/tes-data not found!${NC}"
    echo "Please run: python generate_test_data.py first"
    exit 1
fi

# Ensure base image exists
echo -e "${YELLOW}Checking base image appfl/client:latest...${NC}"
if ! docker image inspect appfl/client:latest >/dev/null 2>&1; then
    echo -e "${RED}Error: Base image appfl/client:latest not found!${NC}"
    echo "Please build it first with: docker build -t appfl/client:latest ."
    exit 1
fi

echo -e "${GREEN}Base image found${NC}"

# Copy data into build context
echo -e "${YELLOW}Copying data into build context...${NC}"
cp -r /tmp/tes-data ./

# Build client 1 image
echo -e "${YELLOW}Building appfl/client1:data-embedded...${NC}"
docker build -f Dockerfile.client1 -t appfl/client1:data-embedded ../../../tes \
    --build-arg CLIENT_ID=TESClient1

# Build client 2 image
echo -e "${YELLOW}Building appfl/client2:data-embedded...${NC}"
docker build -f Dockerfile.client2 -t appfl/client2:data-embedded ../../../tes \
    --build-arg CLIENT_ID=TESClient2

# Clean up copied data from build context
echo -e "${YELLOW}Cleaning up build context...${NC}"
rm -rf ./tes-data

echo -e "${GREEN}✅ Client images built successfully!${NC}"
echo ""
echo "Created images:"
echo "  🐳 appfl/client1:data-embedded (with client_0 data)"
echo "  🐳 appfl/client2:data-embedded (with client_1 data)"
echo ""
echo "To test data loading:"
echo "  docker run --rm appfl/client1:data-embedded ls -la /data"
echo "  docker run --rm appfl/client2:data-embedded ls -la /data"
echo ""
echo "Update your client configs to use these images:"
echo "  TESClient1: docker_image: appfl/client1:data-embedded"
echo "  TESClient2: docker_image: appfl/client2:data-embedded"
