# -------- build stage --------
FROM node:20-alpine AS build
WORKDIR /app

ARG VITE_BACKEND=http://localhost:8000
ENV VITE_BACKEND=${VITE_BACKEND}

COPY package*.json tsconfig.json vite.config.ts index.html tailwind.config.js ./

RUN npm install

COPY src ./src

RUN npm run build

# -------- run stage --------
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
