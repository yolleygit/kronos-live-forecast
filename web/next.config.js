/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove output: 'export' for development - we'll add it back for production builds
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Remove asset prefix for development
  // assetPrefix: process.env.NODE_ENV === 'production' ? '/web' : '',
  // basePath: process.env.NODE_ENV === 'production' ? '/web' : ''
}

module.exports = nextConfig