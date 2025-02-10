FROM denoland/deno:alpine

WORKDIR /app

COPY .env.local /app/.env.local

COPY deno.json /app/deno.json

COPY src /app/src

RUN chown -R deno:deno /app

USER deno

RUN deno cache src/main.ts

WORKDIR /app

CMD ["deno", "run", "--allow-net", "--allow-read", "--allow-env", "--allow-sys", "src/main.ts"]