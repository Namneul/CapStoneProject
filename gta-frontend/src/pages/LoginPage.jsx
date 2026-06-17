import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, Mail, UserPlus } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Layout } from '../components/layout/Layout';

export const LoginPage = () => {
  const navigate = useNavigate();
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
  });

  const updateField = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    navigate('/setup');
  };

  const isSignup = mode === 'signup';

  return (
    <Layout>
      <div className="fade-in w-full max-w-md">
        <Card className="p-10 border border-slate-100">
          <div className="text-center mb-8">
            <h1 className="text-[2.5rem] font-black text-primary mb-2 tracking-tight">G.T.A</h1>
            <p className="text-slate-500 font-medium text-sm">
              면접 답변과 비언어 신호를 함께 분석하는 연습 시스템
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2 rounded-2xl bg-slate-100 p-1 mb-6">
            <button
              type="button"
              onClick={() => setMode('login')}
              className={`rounded-xl py-3 text-sm font-black transition-all ${
                mode === 'login' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
              }`}
            >
              로그인
            </button>
            <button
              type="button"
              onClick={() => setMode('signup')}
              className={`rounded-xl py-3 text-sm font-black transition-all ${
                mode === 'signup' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'
              }`}
            >
              회원가입
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignup && (
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <UserPlus className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors" />
                </div>
                <input
                  type="text"
                  value={form.name}
                  onChange={(event) => updateField('name', event.target.value)}
                  placeholder="이름"
                  className="w-full pl-12 pr-4 py-4 bg-background border border-slate-200 rounded-2xl focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none text-slate-700 placeholder:text-slate-400 font-medium"
                />
              </div>
            )}

            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Mail className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors" />
              </div>
              <input
                type="text"
                value={form.email}
                onChange={(event) => updateField('email', event.target.value)}
                placeholder="아이디 또는 이메일"
                className="w-full pl-12 pr-4 py-4 bg-background border border-slate-200 rounded-2xl focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none text-slate-700 placeholder:text-slate-400 font-medium"
              />
            </div>

            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-slate-400 group-focus-within:text-primary transition-colors" />
              </div>
              <input
                type="password"
                value={form.password}
                onChange={(event) => updateField('password', event.target.value)}
                placeholder="비밀번호"
                className="w-full pl-12 pr-4 py-4 bg-background border border-slate-200 rounded-2xl focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none text-slate-700 placeholder:text-slate-400 font-medium"
              />
            </div>

            <Button type="submit" className="w-full py-4 text-lg">
              {isSignup ? '회원가입하고 시작하기' : '로그인하고 시작하기'}
            </Button>
          </form>
        </Card>
      </div>
    </Layout>
  );
};
