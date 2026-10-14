import { CorpusLab, Playground, TfidfManual } from '../StudiKasus09';
import ToolShell from './ToolShell';

export default function TextTool() {
  return (
    <ToolShell title="Text Mining" cases={['09']}
      intro="Olah teks berbahasa Indonesia: preprocessing bertahap (filtering, case folding, tokenizing, stopword dengan negasi dipertahankan, normalisasi slang, stemming Sastrawi), kalkulator TF-IDF manual, sentimen leksikon InSet, klasifikasi teks, pemodelan topik LDA/NMF, dan ko-okurensi — pada kolom teks CSV Anda.">
      <div className="space-y-5"><Playground /><TfidfManual /><CorpusLab generic /></div>
    </ToolShell>
  );
}
