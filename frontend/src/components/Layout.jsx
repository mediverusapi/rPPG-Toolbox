import React from 'react'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b bg-white">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
          <h1 className="font-semibold">rPPG Toolbox</h1>
        </div>
      </header>
      <main className="flex-1">
        <div className="max-w-5xl mx-auto p-4">
          {children}
        </div>
      </main>
      <footer className="text-xs text-gray-500 py-4 text-center">Local demo UI</footer>
    </div>
  )
}

