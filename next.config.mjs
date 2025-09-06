/** @type {import('next').NextConfig} */
import createJiti from "jiti";
import { fileURLToPath } from "node:url";

const jiti = createJiti(fileURLToPath(import.meta.url));
jiti("./src/env/client");

const nextConfig = {};

export default nextConfig;
