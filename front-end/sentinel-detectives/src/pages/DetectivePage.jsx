import { useState, useRef } from 'react';
import { ArrowLeft, Upload, AlertTriangle, CheckCircle, FileText, Camera, X, Headphones } from 'lucide-react';
import { Link } from 'react-router-dom';

const DetectivePage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadedFile(file);
    
    // Create a preview URL for the uploaded image/video/audio
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setUploadedFile(file);
      
      // Create a preview URL for the uploaded file
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
  };
  
  const handleAnalyze = () => {
    if (!uploadedFile) return;
    
    setIsAnalyzing(true);
    
    // Simulate analysis time
    setTimeout(() => {
      setIsAnalyzing(false);
      setShowResults(true);
    }, 3000);
  };
  
  const resetForm = () => {
    setShowResults(false);
    setUploadedFile(null);
    setPreviewUrl(null);
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Determine if file is audio
  const isAudio = uploadedFile?.type?.startsWith('audio/');

  return (
    <div className="pt-24 pb-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        {/* Header Section */}
        <div className="flex items-center gap-3 mb-8">
          <Link to="/" className="text-gray-500 hover:text-teal-600 transition-colors">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-3xl font-bold text-gray-800">Hercule Poirot</h1>
          <span className="bg-teal-100 text-teal-800 text-xs font-medium px-2.5 py-0.5 rounded">Deepfake Detector</span>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* Left Column - Sentinel Info */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-6 sticky top-24">
              <div className="w-20 h-20 mx-auto bg-gradient-to-b from-teal-500 to-teal-700 rounded-full flex items-center justify-center mb-4 shadow-lg">
                <div className="relative">
                  <Camera size={32} className="text-white" />
                  <Headphones size={18} className="text-white absolute -bottom-1 -right-1" />
                </div>
              </div>
              
              <h2 className="text-xl font-bold text-center text-gray-800 mb-2">Hercule Poirot</h2>
              <p className="text-center text-teal-600 font-medium text-sm mb-4">Deepfake Detector</p>
              
              <div className="border-t border-gray-100 pt-4 mt-2">
                <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                  <div className="relative">
                    <Camera size={16} />
                    <Headphones size={10} className="absolute -bottom-1 -right-1" />
                  </div>
                  <span>How Hercule Poirot Helps You</span>
                </h3>
                <ul className="text-gray-600 text-sm space-y-2">
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Identifies manipulated images and videos</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Detects inconsistencies in facial features</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Analyzes digital artifacts and anomalies</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Detects AI-generated or manipulated audio</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Identifies voice cloning and audio splicing</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* Right Column - Main Content */}
          <div className="lg:col-span-2">
            {!isAnalyzing && !showResults ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-teal-600 to-teal-800 p-6 text-white">
                  <h2 className="text-xl font-bold mb-2">Deepfake Detector</h2>
                  <p className="opacity-90">Upload an image, video, or audio file to detect potential manipulations</p>
                </div>
                
                <div className="p-6">
                  <div 
                    className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center transition-colors ${
                      previewUrl ? 'border-teal-300 bg-teal-50' : 'border-gray-300 hover:border-teal-300 hover:bg-gray-50'
                    }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onClick={triggerFileInput}
                    style={{ minHeight: '200px', cursor: 'pointer' }}
                  >
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      accept="image/*,video/*,audio/*" 
                      className="hidden" 
                    />
                    
                    {previewUrl ? (
                      <div className="w-full flex flex-col items-center">
                        <div className="relative mb-4">
                          {uploadedFile.type.startsWith('image/') ? (
                            <img 
                              src={previewUrl} 
                              alt="Preview" 
                              className="max-h-48 max-w-full rounded shadow-sm" 
                            />
                          ) : uploadedFile.type.startsWith('video/') ? (
                            <video 
                              src={previewUrl} 
                              className="max-h-48 max-w-full rounded shadow-sm" 
                              controls 
                            />
                          ) : (
                            <div className="w-full max-w-sm h-48 flex items-center justify-center bg-gray-800 rounded shadow-sm">
                              <div className="flex flex-col items-center">
                                <Headphones size={48} className="text-teal-400 mb-2" />
                                <audio src={previewUrl} controls className="w-64" />
                              </div>
                            </div>
                          )}
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              resetForm();
                            }}
                            className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-md hover:bg-red-600 transition-colors"
                          >
                            <X size={16} />
                          </button>
                        </div>
                        <p className="text-gray-700 font-medium">{uploadedFile.name}</p>
                        <p className="text-gray-500 text-sm">{(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center space-x-2 mb-3">
                          <Camera size={36} className="text-gray-400" />
                          <Headphones size={28} className="text-gray-400" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-700 mb-1">Drop your file here</h3>
                        <p className="text-gray-500 text-sm mb-2">or click to browse</p>
                        <p className="text-gray-400 text-xs">Supports JPG, PNG, MP4, MP3, WAV files</p>
                      </>
                    )}
                  </div>
                  
                  <button
                    onClick={handleAnalyze}
                    disabled={!uploadedFile}
                    className={`mt-4 w-full py-2.5 px-4 rounded-md flex items-center justify-center gap-2 font-medium ${
                      uploadedFile 
                        ? 'bg-gradient-to-r from-teal-600 to-teal-800 text-white hover:from-teal-700 hover:to-teal-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5' 
                        : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                    }`}
                  >
                    {isAudio ? <Headphones size={18} /> : <Camera size={18} />}
                    Analyze Media
                  </button>
                  
                  <div className="mt-6 bg-teal-50 border border-teal-100 rounded-md p-4">
                    <h3 className="text-sm font-semibold text-teal-800 mb-2 flex items-center gap-2">
                      <AlertTriangle size={16} />
                      <span>How Hercule Poirot Works</span>
                    </h3>
                    <ol className="text-sm text-teal-700 space-y-1 list-decimal pl-5">
                      <li>Upload an image, video, or audio file you want to analyze</li>
                      <li>Our AI examines visual features, audio patterns, and digital artifacts</li>
                      <li>For images/videos: facial features, lighting, and manipulation signals</li>
                      <li>For audio: voice patterns, unnatural transitions, and generation markers</li>
                      <li>Review the detailed analysis and conclusion about authenticity</li>
                    </ol>
                  </div>
                </div>
              </div>
            ) : isAnalyzing ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-8 flex flex-col items-center justify-center min-h-[400px]">
                <div className="animate-pulse">
                  <div className="w-20 h-20 bg-gradient-to-b from-teal-500 to-teal-700 rounded-full flex items-center justify-center shadow-lg">
                    <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                  </div>
                </div>
                
                <h3 className="mt-6 text-xl font-bold text-gray-800">Analyzing Media</h3>
                <p className="mt-2 text-gray-600 max-w-md text-center">
                  Hercule Poirot is using his legendary detective skills to examine your {isAudio ? 'audio' : 'media'} for signs of manipulation. This may take a moment...
                </p>
                
                <div className="mt-8 w-full max-w-md bg-gray-200 rounded-full h-2.5">
                  <div className="bg-teal-600 h-2.5 rounded-full animate-pulse w-[70%]"></div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-teal-600 to-teal-800 p-6 text-white flex justify-between items-center">
                  <div>
                    <h2 className="text-xl font-bold mb-1">Analysis Results</h2>
                    <p className="opacity-90 text-sm">Hercule Poirot's deepfake analysis</p>
                  </div>
                  <button
                    onClick={resetForm}
                    className="bg-white/20 hover:bg-white/30 text-white font-medium py-1.5 px-3 rounded-md transition-colors text-sm flex items-center gap-1.5"
                  >
                    <ArrowLeft size={16} />
                    Back
                  </button>
                </div>
                
                <div className="p-6">
                  <div className="flex flex-col md:flex-row gap-6 mb-6">
                    <div className="md:w-1/2">
                      {uploadedFile && previewUrl && (
                        <div className="bg-black rounded-lg overflow-hidden shadow-md">
                          {uploadedFile.type.startsWith('image/') ? (
                            <img 
                              src={previewUrl} 
                              alt="Analyzed media" 
                              className="w-full h-auto object-contain" 
                            />
                          ) : uploadedFile.type.startsWith('video/') ? (
                            <video 
                              src={previewUrl} 
                              className="w-full h-auto object-contain" 
                              controls 
                            />
                          ) : (
                            <div className="w-full h-48 flex items-center justify-center">
                              <div className="flex flex-col items-center">
                                <Headphones size={48} className="text-teal-400 mb-2" />
                                <audio src={previewUrl} controls className="w-64" />
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    
                    <div className="md:w-1/2">
                      <div className="mb-4 text-center">
                        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-1">Analysis Result</h3>
                        <div className="flex items-center justify-center gap-3">
                          <div className="text-4xl font-bold text-red-600">76%</div>
                          <div className="text-lg font-medium text-red-600">Likely Manipulated</div>
                        </div>
                      </div>
                      
                      <div className="bg-red-50 border border-red-100 rounded-md p-4 mb-4">
                        <div className="flex items-start gap-2">
                          <AlertTriangle size={20} className="text-red-600 shrink-0 mt-0.5" />
                          <div>
                            <h4 className="font-medium text-red-800">Manipulation Detected</h4>
                            <p className="text-sm text-red-700 mt-1">
                              {isAudio 
                                ? "This audio has several indicators of digital manipulation, particularly in voice patterns and transitions."
                                : "This media has several indicators of digital manipulation, particularly in facial features."
                              }
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-800 mb-2">File Information</h4>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="text-gray-600">File Name:</div>
                          <div className="text-gray-800 font-medium truncate">{uploadedFile?.name}</div>
                          
                          <div className="text-gray-600">File Size:</div>
                          <div className="text-gray-800 font-medium">{(uploadedFile?.size / (1024 * 1024)).toFixed(2)} MB</div>
                          
                          <div className="text-gray-600">File Type:</div>
                          <div className="text-gray-800 font-medium">{uploadedFile?.type}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Detection Details</h3>
                    <div className="space-y-3">
                      {isAudio ? (
                        <>
                          <DetailItem 
                            title="Unnatural voice transitions detected" 
                            confidence={92}
                            description="Analysis shows abrupt changes in voice timbre and quality that are inconsistent with natural speech patterns."
                          />
                          
                          <DetailItem 
                            title="Neural network generation artifacts present" 
                            confidence={85}
                            description="Spectral analysis reveals patterns consistent with AI voice synthesis, particularly in consonant sounds."
                          />
                          
                          <DetailItem 
                            title="Inconsistent room acoustics" 
                            confidence={79}
                            description="Background acoustic properties change unnaturally between segments, indicating spliced recordings."
                          />
                          
                          <DetailItem 
                            title="Unnatural breathing patterns" 
                            confidence={68}
                            description="Breathing sounds and pauses do not follow natural human speech patterns."
                          />
                        </>
                      ) : (
                        <>
                          <DetailItem 
                            title="Inconsistent shadows around facial features" 
                            confidence={92}
                            description="Shadows around the eyes and nose don't match the main light source in the image."
                          />
                          
                          <DetailItem 
                            title="Unnatural eye blinking patterns" 
                            confidence={85}
                            description="Analysis of video frames shows abnormal blinking frequency and duration."
                          />
                          
                          <DetailItem 
                            title="Digital artifacts present in hair boundaries" 
                            confidence={78}
                            description="Pixel inconsistencies detected along hair edges, indicating poor blending of manipulated elements."
                          />
                          
                          <DetailItem 
                            title="Facial proportions abnormalities" 
                            confidence={70}
                            description="Subtle inconsistencies in facial feature ratios compared to anatomical norms."
                          />
                        </>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Detection Technology</h3>
                    <p className="text-gray-700 mb-3">
                      This analysis uses multiple detection methods to identify potential manipulations:
                    </p>
                    <ul className="text-gray-700 space-y-1 list-disc pl-5">
                      {isAudio ? (
                        <>
                          <li>Voice pattern and cadence analysis</li>
                          <li>Neural network generation artifact detection</li>
                          <li>Acoustic environment consistency checking</li>
                          <li>Natural speech pattern verification</li>
                          <li>Spectrogram analysis for manipulation markers</li>
                        </>
                      ) : (
                        <>
                          <li>Facial geometry and proportion analysis</li>
                          <li>Shadow and lighting consistency detection</li>
                          <li>Skin texture and pore pattern matching</li>
                          <li>Motion and natural movement analysis</li>
                          <li>Digital artifact and compression anomaly detection</li>
                        </>
                      )}
                    </ul>
                  </div>
                  
                  <button
                    className="mt-6 bg-gradient-to-r from-teal-600 to-teal-800 text-white font-medium py-2 px-4 rounded-md hover:from-teal-700 hover:to-teal-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 w-full flex items-center justify-center gap-2"
                  >
                    <span>Download Detailed Report</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const DetailItem = ({ title, confidence, description }) => {
  return (
    <div className="border border-gray-100 rounded-md p-3 bg-gray-50">
      <div className="flex justify-between items-start mb-1">
        <h4 className="font-medium text-gray-800">{title}</h4>
        <span className="text-xs font-semibold bg-red-100 text-red-800 py-0.5 px-2 rounded-full">
          {confidence}% confidence
        </span>
      </div>
      <p className="text-sm text-gray-600">{description}</p>
    </div>
  );
};

export default DetectivePage;