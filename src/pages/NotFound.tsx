import { Link } from 'react-router-dom';
export default function NotFound() {
  return <div className="card space-y-2"><h1 className="text-4xl font-bold">404</h1><p>Halaman tidak ditemukan.</p><Link to="/">Kembali ke Beranda</Link></div>;
}
