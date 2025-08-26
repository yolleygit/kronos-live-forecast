import type { Metadata } from 'next'
import { Inter, Roboto_Slab } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const robotoSlab = Roboto_Slab({ subsets: ['latin'], variable: '--font-roboto-slab' })

export const metadata: Metadata = {
  title: 'Kronos Live Forecast | BTC/USDT',
  description: 'Real-time cryptocurrency price prediction using Kronos foundation model',
  keywords: ['cryptocurrency', 'bitcoin', 'prediction', 'AI', 'machine learning', 'trading'],
  authors: [{ name: 'The Kronos Project' }],
  viewport: 'width=device-width, initial-scale=1',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${robotoSlab.variable}`}>
      <body className={`${inter.className} antialiased min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800`}>
        {children}
      </body>
    </html>
  )
}