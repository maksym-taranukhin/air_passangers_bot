import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'Air Passenger Rights Chatbot (DEMO)',
  description: 'Chatbot that answers questions about air passenger rights',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full`}>
        <div className="flex flex-col h-full md:p-8 bg-white">
          {children}
        </div>
      </body>
    </html>
  );
}
