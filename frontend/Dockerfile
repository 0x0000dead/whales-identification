# -------- build stage --------
FROM node:20-alpine AS build
WORKDIR /app

# JS-манифест + конфиги + index.html
COPY package*.json tsconfig.json vite.config.ts index.html ./

RUN npm install

# исходники
COPY src ./src

RUN npm run build   # dist/ появится здесь

# -------- run stage --------
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
