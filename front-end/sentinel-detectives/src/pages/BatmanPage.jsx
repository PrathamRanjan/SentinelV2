import { useState } from 'react';
import { ArrowLeft, Search, AlertTriangle, CheckCircle, Youtube } from 'lucide-react';
import { Link } from 'react-router-dom';

const BatmanPage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [youtubeUrl, setYoutubeUrl] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!youtubeUrl) return;
    
    setIsAnalyzing(true);
    
    // Simulate analysis time
    setTimeout(() => {
      setIsAnalyzing(false);
      setShowResults(true);
    }, 3000);
  };
  
  const resetForm = () => {
    setShowResults(false);
    setYoutubeUrl('');
  };

  return (
    <div className="pt-24 pb-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        {/* Header Section */}
        <div className="flex items-center gap-3 mb-8">
          <Link to="/" className="text-gray-500 hover:text-cyan-600 transition-colors">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-3xl font-bold text-gray-800">Batman</h1>
          <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded">YouTube Fact Checker</span>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* Left Column - Sentinel Info */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-6 sticky top-24">
              <div className="w-20 h-20 mx-auto bg-gradient-to-b from-blue-500 to-blue-700 rounded-full flex items-center justify-center mb-4 shadow-lg">
                <Youtube size={32} className="text-white" />
              </div>
              
              <h2 className="text-xl font-bold text-center text-gray-800 mb-2">Batman</h2>
              <p className="text-center text-cyan-600 font-medium text-sm mb-4">YouTube Fact Checker</p>
              
              <div className="border-t border-gray-100 pt-4 mt-2">
                <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                  <Search size={16} />
                  <span>How Batman Helps You</span>
                </h3>
                <ul className="text-gray-600 text-sm space-y-2">
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Analyzes YouTube videos for factual accuracy</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Identifies misleading claims in video content</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Provides detailed analysis with evidence</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle size={16} className="text-green-500 mt-0.5 shrink-0" />
                    <span>Offers a trust score based on content analysis</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* Right Column - Main Content */}
          <div className="lg:col-span-2">
            {!isAnalyzing && !showResults ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-6 text-white">
                  <h2 className="text-xl font-bold mb-2">YouTube Fact Checker</h2>
                  <p className="opacity-90">Enter a YouTube URL to analyze the content for factual accuracy</p>
                </div>
                
                <div className="p-6">
                  <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                      <label htmlFor="youtube-url" className="block text-sm font-medium text-gray-700 mb-1">
                        YouTube Video URL
                      </label>
                      <div className="flex">
                        <input
                          id="youtube-url"
                          type="text"
                          className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-cyan-300 focus:border-cyan-500 transition-all duration-300"
                          placeholder="https://www.youtube.com/watch?v=..."
                          value={youtubeUrl}
                          onChange={(e) => setYoutubeUrl(e.target.value)}
                        />
                      </div>
                    </div>
                    
                    <button
                      type="submit"
                      className="bg-gradient-to-r from-cyan-600 to-cyan-800 text-white font-medium py-2 px-4 rounded-md hover:from-cyan-700 hover:to-cyan-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 w-full flex items-center justify-center gap-2"
                    >
                      <Search size={18} />
                      Analyze Video
                    </button>
                  </form>
                  
                  <div className="mt-6 bg-blue-50 border border-blue-100 rounded-md p-4">
                    <h3 className="text-sm font-semibold text-blue-800 mb-2 flex items-center gap-2">
                      <AlertTriangle size={16} />
                      <span>How to Use Batman's Analysis</span>
                    </h3>
                    <ol className="text-sm text-blue-700 space-y-1 list-decimal pl-5">
                      <li>Paste a full YouTube URL in the field above</li>
                      <li>Click "Analyze Video" to start the fact-checking process</li>
                      <li>Review the detailed analysis and fact check report</li>
                      <li>Use the trust score to evaluate the content's reliability</li>
                    </ol>
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                      <span>Recent Analyses</span>
                    </h3>
                    
                    <div className="space-y-3">
                      <div className="p-3 border border-gray-100 rounded-md hover:bg-gray-50 transition-colors cursor-pointer">
                        <div className="text-sm font-medium text-gray-800 mb-1 truncate">How AI Will Change Everything</div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-gray-500">Today, 2:45 PM</span>
                          <span className="text-xs font-semibold bg-green-100 text-green-800 py-0.5 px-2 rounded-full">8.4/10</span>
                        </div>
                      </div>
                      
                      <div className="p-3 border border-gray-100 rounded-md hover:bg-gray-50 transition-colors cursor-pointer">
                        <div className="text-sm font-medium text-gray-800 mb-1 truncate">The Truth About Quantum Computing</div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-gray-500">Yesterday, 11:20 AM</span>
                          <span className="text-xs font-semibold bg-red-100 text-red-800 py-0.5 px-2 rounded-full">2.1/10</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : isAnalyzing ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-8 flex flex-col items-center justify-center min-h-[400px]">
                <div className="animate-pulse">
                  <div className="w-20 h-20 bg-gradient-to-b from-blue-500 to-blue-700 rounded-full flex items-center justify-center shadow-lg">
                    <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                  </div>
                </div>
                
                <h3 className="mt-6 text-xl font-bold text-gray-800">Analyzing Video Content</h3>
                <p className="mt-2 text-gray-600 max-w-md text-center">
                  Batman is using his detective skills to analyze the video content for factual accuracy. This may take a moment...
                </p>
                
                <div className="mt-8 w-full max-w-md bg-gray-200 rounded-full h-2.5">
                  <div className="bg-blue-600 h-2.5 rounded-full animate-pulse w-[45%]"></div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-6 text-white flex justify-between items-center">
                  <div>
                    <h2 className="text-xl font-bold mb-1">Analysis Results</h2>
                    <p className="opacity-90 text-sm">Batman's fact-checking report</p>
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
                  <div className="mb-6 flex flex-col items-center">
                    <div className="text-center mb-2">
                      <span className="text-xs font-semibold text-gray-500">TRUST SCORE</span>
                      <div className="flex items-center justify-center gap-2 mt-1">
                        <div className="text-4xl font-bold text-blue-600">6.8</div>
                        <div className="text-gray-400 font-medium">/10</div>
                      </div>
                    </div>
                    
                    <div className="w-full max-w-sm bg-gray-200 rounded-full h-3 mt-2">
                      <div className="bg-blue-600 h-3 rounded-full" style={{ width: '68%' }}></div>
                    </div>
                    
                    <p className="mt-3 text-sm text-gray-600 text-center">
                      This video contains some factual information but also includes several misleading claims and unverified statements.
                    </p>
                  </div>
                  
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Summary</h3>
                    <div className="bg-gray-50 border border-gray-100 rounded-md p-4 text-gray-700">
                      <p>
                        This video about technology advancements contains a mix of accurate and misleading information. 
                        While the presenter correctly describes current AI capabilities, several future predictions are 
                        exaggerated beyond what is supported by current research. The video also omits important 
                        context about technology limitations.
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Fact Check Report</h3>
                    <div className="space-y-3">
                      <div className="border border-green-100 bg-green-50 rounded-md p-3">
                        <div className="flex gap-2">
                          <CheckCircle size={20} className="text-green-600 shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium text-green-800">Current AI systems can analyze images and text with high accuracy</p>
                            <p className="text-sm text-green-700 mt-1">
                              VERIFIED: Modern AI systems achieve human-level or better performance in specific image and text analysis tasks.
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="border border-red-100 bg-red-50 rounded-md p-3">
                        <div className="flex gap-2">
                          <AlertTriangle size={20} className="text-red-600 shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium text-red-800">AI will replace 80% of all jobs within 5 years</p>
                            <p className="text-sm text-red-700 mt-1">
                              FALSE: Research from multiple economic institutions predicts automation impact on jobs to be gradual, 
                              with estimates ranging from 10-30% of tasks (not jobs) being automated over the next decade.
                            </p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="border border-yellow-100 bg-yellow-50 rounded-md p-3">
                        <div className="flex gap-2">
                          <AlertTriangle size={20} className="text-yellow-600 shrink-0 mt-0.5" />
                          <div>
                            <p className="font-medium text-yellow-800">Quantum computing will make current encryption obsolete</p>
                            <p className="text-sm text-yellow-700 mt-1">
                              PARTIALLY TRUE: While quantum computers could theoretically break some current encryption methods, 
                              practical quantum computers with sufficient qubits are still years away, and new quantum-resistant 
                              encryption methods are being developed.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Sources</h3>
                    <ul className="space-y-1">
                      <li className="flex items-start gap-2">
                        <div className="min-w-1.5 h-1.5 rounded-full bg-blue-500 mt-2"></div>
                        <a href="#" className="text-blue-600 hover:underline text-sm">MIT Technology Review: State of AI Report 2024</a>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="min-w-1.5 h-1.5 rounded-full bg-blue-500 mt-2"></div>
                        <a href="#" className="text-blue-600 hover:underline text-sm">McKinsey Global Institute: Jobs Lost, Jobs Gained</a>
                      </li>
                      <li className="flex items-start gap-2">
                        <div className="min-w-1.5 h-1.5 rounded-full bg-blue-500 mt-2"></div>
                        <a href="#" className="text-blue-600 hover:underline text-sm">NIST: Post-Quantum Cryptography</a>
                      </li>
                    </ul>
                  </div>
                  
                  <button
                    className="mt-6 bg-gradient-to-r from-cyan-600 to-cyan-800 text-white font-medium py-2 px-4 rounded-md hover:from-cyan-700 hover:to-cyan-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 w-full flex items-center justify-center gap-2"
                  >
                    <span>Download Full Report</span>
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

export default BatmanPage;