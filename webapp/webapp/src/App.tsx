import { Navbar } from "./components/Navbar";
import { DocuemntParser } from "./components/DocumentParser";

export default function App() {
  return (
    <div className="bg-neutral-100 min-h-screen flex flex-col items-center">
      <div className="relative w-[80%] max-w-[1900px] mt-12">
        <div className="absolute z-index-0 inset-0 translate-x-8 -translate-y-8 rounded-[40px] shadow-lg bg-stone-200" />
        <div className="absolute z-index-0 inset-0 translate-x-4 -translate-y-4 rounded-[40px] shadow-lg bg-stone-100" />

        <div
          className="relative z-index-10 px-2 py-24 rounded-[40px]
       flex flex-col bg-white items-center justify-center shadow-lg"
        >
          <Navbar />

          <h1 className="text-6xl tracking-wider uppercase font-bold">
            <span className="text-violet-900">Vis</span>onary
          </h1>
          <div
            className="mt-4 rounded-full w-64 h-1 bg-gradient-to-r from-indigo-700
           to-violet-900"
          />
          <div className="w-[70%]">
            <p className="mt-12 text-neutral-500 text-lg text-pretty text-center">
              Visionary transforms document images into structured, searchable
              data using OCR and AI-powered extraction. Upload invoices, forms,
              or scanned documents, and Visionary automatically detects,
              organizes, and stores key information in a clean database-ready
              format.
            </p>
          </div>
            <a href="#parser" className="mt-4 text-xl 
            tracking-wide bg-gradient-to-r from-indigo-500 to-violet-600 px-6
             py-1 rounded-full text-white
             hover:from-indigo-400 hover:to-violet-500
             transition-colors duration-300 cursor-pointer"> Try it out</a>

        </div>

      </div>

      <div className="w-[80%] mt-12 shadow-lg">
        <DocuemntParser />
      </div>
    </div>
  );
}
