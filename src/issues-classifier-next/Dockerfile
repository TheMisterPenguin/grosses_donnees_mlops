FROM node:23-alpine AS builder

WORKDIR /app

COPY package.json yarn.lock ./

RUN yarn install

COPY ./ ./

RUN yarn build

FROM node:23-alpine

WORKDIR /app

COPY --from=builder /app/package.json /app/yarn.lock ./

RUN yarn install --production

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.ts ./next.config.ts

EXPOSE 3000

CMD [ "yarn", "start" ]