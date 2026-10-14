import { lazy, Suspense } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import { Spinner } from './components/ui';
import { LangProvider } from './lib/i18n';
import Home from './pages/Home';

const T = {
  t01: lazy(() => import('./pages/Tugas01')), t02: lazy(() => import('./pages/Tugas02')), t03: lazy(() => import('./pages/Tugas03')),
  t04: lazy(() => import('./pages/Tugas04')), t05: lazy(() => import('./pages/Tugas05')), t06: lazy(() => import('./pages/Tugas06')),
  t07: lazy(() => import('./pages/Tugas07')), t08: lazy(() => import('./pages/Tugas08')), t09: lazy(() => import('./pages/Tugas09')),
};
const RegressionLab = lazy(() => import('./pages/RegressionLab'));
const DataLab = lazy(() => import('./pages/DataLab'));
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
              <Route path="/tugas-01" element={<T.t01 />} />
              <Route path="/tugas-02" element={<T.t02 />} />
              <Route path="/tugas-03" element={<T.t03 />} />
              <Route path="/tugas-04" element={<T.t04 />} />
              <Route path="/tugas-05" element={<T.t05 />} />
              <Route path="/tugas-06" element={<T.t06 />} />
              <Route path="/tugas-07" element={<T.t07 />} />
              <Route path="/tugas-08" element={<T.t08 />} />
              <Route path="/tugas-09" element={<T.t09 />} />
              <Route path="/regression-lab" element={<RegressionLab />} />
              <Route path="/data-lab" element={<DataLab />} />
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
