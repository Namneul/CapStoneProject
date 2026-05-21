import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Lock } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Modal } from '../components/ui/Modal';
import { Layout } from '../components/layout/Layout';

export const LoginPage = () => {
  const navigate = useNavigate();
  const [showModal, setShowModal] = useState(false);

  return (
    <Layout>
      <div className="fade-in w-full max-w-md">
        <Card className="p-10 border border-slate-100">
          <div className="text-center mb-10">
            <h1 className="text-[2.5rem] font-black text-primary mb-2 tracking-tight">G.T.A</h1>
            <p className="text-slate-500 font-medium text-sm">멀티에이전트 표현 습관 분석 시스템</p>
          </div>

          <div className="space-y-4 mb-8">
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <User className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors" />
              </div>
              <input type="email" placeholder="이메일 주소" className="w-full pl-12 pr-4 py-4 bg-background border border-slate-200 rounded-2xl focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none text-slate-700 placeholder:text-slate-400 font-medium" />
            </div>
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors" />
              </div>
              <input type="password" placeholder="비밀번호" className="w-full pl-12 pr-4 py-4 bg-background border border-slate-200 rounded-2xl focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none text-slate-700 placeholder:text-slate-400 font-medium" />
            </div>
          </div>

          <Button onClick={() => navigate('/setup')} className="w-full py-4 text-lg">
            로그인
          </Button>

          <div className="text-center mt-6">
            <span className="text-sm text-slate-500 font-medium">계정이 없으신가요? </span>
            <button onClick={() => setShowModal(true)} className="text-sm text-primary font-bold hover:underline ml-1">
              회원가입
            </button>
          </div>
        </Card>

        <Modal isOpen={showModal} onClose={() => setShowModal(false)} title="안내">
          회원가입 기능은 현재 준비 중입니다.<br />기존 계정으로 로그인해주세요.
        </Modal>
      </div>
    </Layout>
  );
};
