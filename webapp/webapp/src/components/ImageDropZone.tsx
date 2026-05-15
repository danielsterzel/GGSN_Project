import { useState } from "react";

export function ImageDropZone() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const allowedTypes = ["image/png", "image/jpeg"];

  function handleFile(selectedFile: File) {
    if (!allowedTypes.includes(selectedFile.type)) {
      alert("Please upload PNG or JPG image");
      return;
    }

    setFile(selectedFile);
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);

        const droppedFile = event.dataTransfer.files[0];

        if (!droppedFile) return;

        handleFile(droppedFile);
      }}
      className={`
        flex min-h-64 w-full max-w-3xl cursor-pointer
        flex-col items-center justify-center
        rounded-3xl border-2 border-dashed
        p-12 text-center transition-all duration-300
        ${
          isDragging
            ? "border-indigo-500 bg-indigo-50"
            : "border-neutral-300 bg-white hover:border-indigo-400"
        }
      `}
    >
      <input
        type="file"
        accept=".png,.jpg,.jpeg"
        className="hidden"
        id="image-upload"
        onChange={(event) => {
          const selectedFile = event.target.files?.[0];

          if (!selectedFile) return;

          handleFile(selectedFile);
        }}
      />

      <label
        htmlFor="image-upload"
        className="flex cursor-pointer flex-col items-center"
      >
        <div className="text-5xl text-indigo-500">
          <i className="fa-solid fa-image"></i>
        </div>

        <h2 className="mt-4 text-2xl font-semibold text-neutral-900">
          Drop your image here
        </h2>

        <p className="mt-2 text-neutral-500">
          PNG and JPG supported
        </p>

        {file && (
          <div>
          <p
            className="
              mt-6 rounded-full
              bg-indigo-50 px-4 py-2
              text-sm font-medium text-indigo-600
            "
          >
            Selected: {file.name}
          </p>
          <button 
            onClick={() => setFile(null)}
            className="mt-4 bg-gradient-to-r from-indigo-400 to-violet-500
            hover:from-indigo-500 hover:to-violet-600 transition-colors
            duration-300 rounded-full px-8 py-1 text-white">
          Clear</button>
          </div>
        )}
      </label>
    </div>
  );
}