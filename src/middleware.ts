import { type NextRequest, NextResponse } from "next/server";

const UNPROTECTED_PATH_PREFIX = ["/auth"];

export default function middleware(request: NextRequest) {
  const response = NextResponse.next();
  const url = request.nextUrl;

  if (
    UNPROTECTED_PATH_PREFIX.some((prefix) => url.pathname.startsWith(prefix))
  ) {
    return response;
  }
  return NextResponse.redirect(new URL("/auth/login", request.url));
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico, sitemap.xml, robots.txt (metadata files)
     * - /auth/login, /auth/forgot-password (unprotected paths)
     */
    "/((?!api|_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
  ],
};
