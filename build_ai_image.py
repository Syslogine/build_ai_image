import os
import subprocess
import requests

def get_latest_cuda_tag():
    api_url = "https://hub.docker.com/v2/repositories/nvidia/cuda/tags"
    response = requests.get(api_url)
    response.raise_for_status()
    tags = response.json()["results"]
    sorted_tags = sorted(tags, key=lambda x: x["last_updated"], reverse=True)
    latest_tag = sorted_tags[0]["name"]
    return latest_tag

def get_user_confirmation():
    confirmation = input("Do you want to proceed with building the Docker image? (y/n) ")
    if confirmation.lower() == "y":
        return True
    else:
        return False

def create_app_directory():
    app_dir = "app"
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
        print(f"Created {app_dir} directory.")
    else:
        print(f"{app_dir} directory already exists.")

# Print initial message
print("Fetching the latest NVIDIA CUDA base image tag...")

# Get the latest NVIDIA CUDA base image tag
BASE_IMAGE_TAG = get_latest_cuda_tag()
BASE_IMAGE = f"nvidia/cuda:{BASE_IMAGE_TAG}"

print(f"Using base image: {BASE_IMAGE}")

TENSORFLOW_VERSION = "2.11.0"
ADDITIONAL_PACKAGES = ["git", "wget", "build-essential"]

# Create app directory
create_app_directory()

# Generate Dockerfile content
additional_packages_str = " ".join(ADDITIONAL_PACKAGES)

# Print the base image to debug
print(f"Base image: {BASE_IMAGE}")

# Determine package manager based on the base image distribution
if "debian" in BASE_IMAGE.lower() or "ubuntu" in BASE_IMAGE.lower():
    package_manager = "apt-get"
    detected_distribution = "Debian/Ubuntu"
elif "centos" in BASE_IMAGE.lower() or "rhel" in BASE_IMAGE.lower() or "ubi" in BASE_IMAGE.lower():
    package_manager = "yum"
    detected_distribution = "CentOS/RHEL"
else:
    detected_distribution = "Unknown"
    raise ValueError("Unsupported base image distribution")

print(f"Detected base image distribution: {detected_distribution}")

# Generate Dockerfile content based on the detected package manager
if package_manager == "apt-get":
    dockerfile_content = f"""\
    FROM {BASE_IMAGE}

    RUN apt-get update && \
        apt-get install -y {additional_packages_str} python3 python3-pip && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --no-cache-dir tensorflow=={TENSORFLOW_VERSION}
    """
elif package_manager == "yum":
    dockerfile_content = f"""\
    FROM {BASE_IMAGE}

    RUN yum install -y git wget redhat-rpm-config python3 python3-devel python3-pip && \
        yum clean all && \
        rm -rf /var/cache/yum/* && \
        pip3 install --no-cache-dir tensorflow
    """
else:
    raise ValueError("Unsupported package manager")

# Write Dockerfile to disk
with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)

print("Dockerfile generated successfully.")

# Get user confirmation
if get_user_confirmation():
    print("Building Docker image...")
    try:
        subprocess.run(["docker", "build", "-t", "my-ai-image", "."], check=True)
        print("Docker image built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Docker build failed with exit code {e.returncode}")
        if e.output:
            print(e.output.decode())
else:
    print("Docker image build canceled.")

# Optional: Push Docker image to registry
# subprocess.run(["docker", "push", "my-ai-image"], check=True)