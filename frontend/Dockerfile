FROM node:20-alpine AS builder
WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY package*.json ./
RUN npm ci

# ── Build args (baked into the Next.js bundle at build time) ──────────────────
# NEXT_PUBLIC_* vars must be available during `npm run build`, not just runtime.
ARG NEXT_PUBLIC_API_URL=http://backend:8000
ARG NEXT_PUBLIC_GOOGLE_MAPS_API_KEY

ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=$NEXT_PUBLIC_GOOGLE_MAPS_API_KEY

# ── Build ─────────────────────────────────────────────────────────────────────
COPY . .
RUN npm run build

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
