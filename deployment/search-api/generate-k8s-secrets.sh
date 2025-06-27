#!/bin/bash

echo "Enter your GitHub Container Registry username:"
read -r GHCR_USERNAME

echo "Enter your GitHub PAT (Personal Access Token with read:packages scope):"
read -rs GHCR_PAT
echo

echo "Enter the Database Host:"
read -r DB_HOST
echo "Enter the Database Port:"
read -r DB_PORT
echo "Enter the Database Name:"
read -r DB_NAME
echo "Enter the Database User:"
read -r DB_USER
echo "Enter the Database Password:"
read -rs DB_PASS
echo

AUTH=$(echo -n "$GHCR_USERNAME:$GHCR_PAT" | base64)
DOCKER_CONFIG_JSON=$(echo -n "{\"auths\":{\"ghcr.io\":{\"username\":\"$GHCR_USERNAME\",\"password\":\"$GHCR_PAT\",\"auth\":\"$AUTH\"}}}" | base64 -w0)

cat <<EOF > search-api.secret.ghcr.yaml
apiVersion: v1
kind: Secret
metadata:
  name: search-api-ghcr-auth
  namespace: athenarc 
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: $DOCKER_CONFIG_JSON
EOF

cat <<EOF > search-api.secret.app.yaml
apiVersion: v1
kind: Secret
metadata:
  name: search-api-app-secrets
  namespace: athenarc
type: Opaque
stringData:
  DB_HOST: "$DB_HOST"
  DB_PORT: "$DB_PORT"
  DB_NAME: "$DB_NAME"
  DB_USER: "$DB_USER"
  DB_PASS: "$DB_PASS"
EOF

echo "✅ Secrets generated:"
echo "  - search-api.secret.ghcr.yaml (image pull secret)"
echo "  - search-api.secret.app.yaml (application secrets)"