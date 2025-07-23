# Stage 1: Build React frontend
FROM node:18 AS client-build

WORKDIR /app

COPY client/package*.json ./client/
RUN cd client && npm install

COPY client ./client
RUN cd client && npm run build

# Stage 2: Build backend and copy frontend build
FROM node:18

WORKDIR /app

# Install backend dependencies
COPY package*.json ./
RUN npm install

# Copy backend source
COPY . .

# Copy React build from previous stage
COPY --from=client-build /app/client/build ./client/build

# Expose backend port
EXPOSE 5000

# Start the backend
CMD ["node", "index.js"]