import { ImageDropZone } from "./ImageDropZone"

export function DocuemntParser()
{
    return (<div id="parser" className="w-full flex flex-col items-center
    bg-white rounded-[40px] p-12">

        <div className="mb-12 flex flex-col items-center justify-center">
            <h1 className="tracking-wider text-5xl uppercase font-bold">
                Document Parser
            </h1>
            <div className="mt-4 w-[80%]">
            <p className="text-neutral-500 text-lg text-pretty text-center">
                Drag and drop a image containing document you want to digitalize
                and extract to a pdf format.
            </p>
            </div>
        </div>
        <ImageDropZone />

        <button 
            className="bg-white mt-4 border border-black rounded-full px-12 py-1
            hover:bg-black/5 transition-colors duration-300 cursor-pointer shadow-lg">
          Extract to PDF
        </button>
    </div>)
}