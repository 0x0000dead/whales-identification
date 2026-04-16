# -------- build stage --------
FROM node:20-slim AS build
WORKDIR /app

ARG VITE_BACKEND=http://localhost:8000
ENV VITE_BACKEND=${VITE_BACKEND}

# Allow overriding npm registry for restricted-network environments.
# Pass --build-arg NPM_REGISTRY=https://registry.npmmirror.com if npmjs.org is blocked.
ARG NPM_REGISTRY=https://registry.npmjs.org

COPY package*.json tsconfig.json vite.config.ts index.html ./

RUN npm config set registry ${NPM_REGISTRY} && npm install

COPY src ./src

RUN node_modules/.bin/vite build

# -------- run stage --------
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
