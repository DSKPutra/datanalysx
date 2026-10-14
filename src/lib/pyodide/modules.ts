/* Sumber modul Python dmx dibundel sebagai teks (Vite ?raw) lalu ditulis ke FS Pyodide. */
import init from '../../../python/dmx/__init__.py?raw';
import classify from '../../../python/dmx/classify.py?raw';
import cluster from '../../../python/dmx/cluster.py?raw';
import common from '../../../python/dmx/common.py?raw';
import datalab from '../../../python/dmx/datalab.py?raw';
import eda from '../../../python/dmx/eda.py?raw';
import t01 from '../../../python/dmx/t01.py?raw';
import t02 from '../../../python/dmx/t02.py?raw';
import t03 from '../../../python/dmx/t03.py?raw';
import t04 from '../../../python/dmx/t04.py?raw';
import t05 from '../../../python/dmx/t05.py?raw';
import t06 from '../../../python/dmx/t06.py?raw';
import t07 from '../../../python/dmx/t07.py?raw';
import t08 from '../../../python/dmx/t08.py?raw';
import t09 from '../../../python/dmx/t09.py?raw';

export const pythonModules: Record<string, string> = {
  '__init__.py': init,
  'common.py': common,
  'classify.py': classify,
  'cluster.py': cluster,
  'eda.py': eda,
  'datalab.py': datalab,
  't01.py': t01,
  't02.py': t02,
  't03.py': t03,
  't04.py': t04,
  't05.py': t05,
  't06.py': t06,
  't07.py': t07,
  't08.py': t08,
  't09.py': t09,
};
