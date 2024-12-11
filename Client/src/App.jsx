import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const ImageInfill = () => {
  const [image, setImage] = useState(null);
  const [maskedImage, setMaskedImage] = useState(null);
  const [resultImages, setResultImages] = useState({});
  const [drawing, setDrawing] = useState(false);
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    ctxRef.current = canvas.getContext('2d');
  }, []);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        setImage(event.target.result);
        const canvas = canvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;
        ctxRef.current.globalAlpha = 0.5;
        ctxRef.current.drawImage(img, 0, 0);
        ctxRef.current.globalAlpha = 1;
      };
      img.src = event.target.result;
    };

    reader.readAsDataURL(file);
  };

  const startDrawing = (e) => {
    setDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setDrawing(false);
    ctxRef.current.beginPath();
  };

  const draw = (e) => {
    if (!drawing) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctxRef.current.lineWidth = 1;
    ctxRef.current.lineCap = 'round';
    ctxRef.current.strokeStyle = 'black';

    ctxRef.current.lineTo(x, y);
    ctxRef.current.stroke();
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(x, y);
  };

  const saveAndProcessImage = async () => {
    if (!image) return;

    const maskCanvas = canvasRef.current.toDataURL('image/png');

    try {
      const response = await axios.post('http://127.0.0.1:5000/inpaint', {
        original: image,
        mask: maskCanvas,
      });

      setResultImages(response.data);
      setMaskedImage(maskCanvas); // Save the user-masked image
    } catch (error) {
      console.error('Error during inpainting:', error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="mb-4 p-2 border border-gray-300 rounded"
      />
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseMove={draw}
        onMouseLeave={stopDrawing}
        className="border border-gray-300 cursor-crosshair"
      />
      {image && (
        <button
          onClick={saveAndProcessImage}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Process Image
        </button>
      )}
      <div className="mt-4 grid grid-cols-2 gap-4">
        {resultImages.originalImage && (
          <div>
            <h4>Original Image</h4>
            <img src={resultImages.originalImage} alt="Original" />
          </div>
        )}
        {maskedImage && (
          <div>
            <h4>Masked by User</h4>
            <img src={maskedImage} alt="Masked by User" />
          </div>
        )}
        {resultImages.pythonGeneratedMask && (
          <div>
            <h4>Python Generated Mask</h4>
            <img src={resultImages.pythonGeneratedMask} alt="Generated Mask" />
          </div>
        )}
        {resultImages.inpaintedImage && (
          <div>
            <h4>Inpainted Image</h4>
            <img src={resultImages.inpaintedImage} alt="Inpainted" />
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageInfill;
