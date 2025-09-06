import axios from "axios";
import { env } from "@/env/client";

const axiosInstance = axios.create({
  baseURL: env.NEXT_PUBLIC_BASE_URL,
});

axiosInstance.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response && error.response.status) {
      throw new Error(error.response.data.message);
    } else {
      console.error("An error occurred:", error);
    }

    return Promise.reject(error);
  }
);

export default axiosInstance;
