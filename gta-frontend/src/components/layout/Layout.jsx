import React from 'react';
import { Header } from './Header';
import { Footer } from './Footer';

/**
 * 기본 레이아웃 컴포넌트
 * (Sidebar는 라우팅 구조에 따라 Result 페이지 등에서만 적용할 수 있도록 확장성을 남겨둠)
 */
export const Layout = ({ children, withHeader = false }) => {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      {withHeader && <Header />}
      <main className="flex-1 flex flex-col items-center justify-center p-4 sm:p-8">
        {children}
      </main>
      <Footer />
    </div>
  );
};
