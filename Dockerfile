# syntax=docker/dockerfile:1

# The estimator is one static HTML file, so the image is nginx and that file.
# nginx-unprivileged runs as the non-root "nginx" user (uid 101) and listens on 8080,
# with its pid file and caches under /tmp, so the container can run with a read-only
# root filesystem (see compose.yaml).
FROM nginxinc/nginx-unprivileged:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz >/dev/null || exit 1
