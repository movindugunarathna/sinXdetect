#!/bin/bash
# Docker entrypoint script for sinXdetect
# Configures nginx based on SSL certificate availability

set -e

SSL_CERT="/etc/letsencrypt/live/api.sinxdetect.movindu.com/fullchain.pem"
SSL_KEY="/etc/letsencrypt/live/api.sinxdetect.movindu.com/privkey.pem"
NGINX_CONF="/etc/nginx/conf.d/sinxdetect.conf"

echo "Checking SSL certificate availability..."

if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
    echo "✅ SSL certificates found. Enabling HTTPS configuration."
    # SSL certificates exist, use the full config with HTTPS
    # The nginx.conf already has SSL configuration, so we're good
else
    echo "⚠️  SSL certificates not found. Using HTTP-only configuration."
    # Create HTTP-only nginx config
    cat > "$NGINX_CONF" << 'NGINX_HTTP_ONLY'
# HTTP-only configuration (SSL certificates not available)

# Frontend server - sinxdetect.movindu.com
server {
    listen 80;
    server_name sinxdetect.movindu.com;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json;

    # Handle client-side routing for SPA
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}

# API server - api.sinxdetect.movindu.com
server {
    listen 80;
    server_name api.sinxdetect.movindu.com;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json;

    # Proxy all requests to backend
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        
        # CORS is handled by FastAPI backend - don't add headers here to avoid duplicates
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}

# Default server for localhost/IP access (optional fallback)
server {
    listen 80 default_server;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json;

    # Proxy API requests to backend (for /api/ path)
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Handle client-side routing for SPA
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
NGINX_HTTP_ONLY
fi

# Test nginx configuration
echo "Testing nginx configuration..."
nginx -t

# Start supervisord
echo "Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
