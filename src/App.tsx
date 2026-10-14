import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import { Spinner } from './components/ui';
import { LangProvider } from './lib/i18n';
import Home from './pages/Home';

const T = {
  t01: lazy(() => import('./pages/StudiKasus01')), t02: lazy(() => import('./pages/StudiKasus02')), t03: lazy(() => import('./pages/StudiKasus03')),
  t04: lazy(() => import('./pages/StudiKasus04')), t05: lazy(() => import('./pages/StudiKasus05')), t06: lazy(() => import('./pages/StudiKasus06')),
  t07: lazy(() => import('./pages/StudiKasus07')), t08: lazy(() => import('./pages/StudiKasus08')), t09: lazy(() => import('./pages/StudiKasus09')),
};
const ExploreTool = lazy(() => import('./pages/tools/ExploreTool'));
const RegressionTool = lazy(() => import('./pages/tools/RegressionTool'));
const ClassificationTool = lazy(() => import('./pages/tools/ClassificationTool'));
const ClusteringTool = lazy(() => import('./pages/tools/ClusteringTool'));
const TextTool = lazy(() => import('./pages/tools/TextTool'));
const Compare = lazy(() => import('./pages/Compare'));
const Lessons = lazy(() => import('./pages/Lessons'));
const About = lazy(() => import('./pages/About'));
const NotFound = lazy(() => import('./pages/NotFound'));

export default function App() {
  const basename = import.meta.env.BASE_URL.replace(/\/$/, '') || '/';
  return (
    <LangProvider>
      <BrowserRouter basename={basename} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <Layout>
          <Suspense fallback={<Spinner text="Memuat halaman …" />}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/studi-kasus-01" element={<T.t01 />} />
              <Route path="/studi-kasus-02" element={<T.t02 />} />
              <Route path="/studi-kasus-03" element={<T.t03 />} />
              <Route path="/studi-kasus-04" element={<T.t04 />} />
              <Route path="/studi-kasus-05" element={<T.t05 />} />
              <Route path="/studi-kasus-06" element={<T.t06 />} />
              <Route path="/studi-kasus-07" element={<T.t07 />} />
              <Route path="/studi-kasus-08" element={<T.t08 />} />
              <Route path="/studi-kasus-09" element={<T.t09 />} />
              <Route path="/eksplorasi" element={<ExploreTool />} />
              <Route path="/regresi" element={<RegressionTool />} />
              <Route path="/klasifikasi" element={<ClassificationTool />} />
              <Route path="/clustering" element={<ClusteringTool />} />
              <Route path="/text-mining" element={<TextTool />} />
              {['01', '02', '03', '04', '05', '06', '07', '08', '09'].map((n) => <Route key={n} path={`/tugas-${n}`} element={<Navigate to={`/studi-kasus-${n}`} replace />} />)}
              <Route path="/regression-lab" element={<Navigate to="/regresi" replace />} />
              <Route path="/data-lab" element={<Navigate to="/klasifikasi" replace />} />
              <Route path="/perbandingan" element={<Compare />} />
              <Route path="/pelajaran" element={<Lessons />} />
              <Route path="/tentang" element={<About />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </LangProvider>
  );
}
