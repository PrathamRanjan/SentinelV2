import { useState } from 'react';
import { ArrowLeft, Gavel, MessageSquare, AlertTriangle, Lightbulb, Scale } from 'lucide-react';
import { Link } from 'react-router-dom';

const CourtroomPage = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [debateInput, setDebateInput] = useState('');
  const [debateResults, setDebateResults] = useState({
    harvey: [],
    saul: []
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!debateInput) return;
    
    setIsAnalyzing(true);
    
    // Simulate analysis time
    setTimeout(() => {
      // Sample debate arguments for demonstration
      setDebateResults({
        harvey: [
          "The evidence clearly supports this position due to three key factors.",
          "First, multiple peer-reviewed studies have demonstrated consistent results.",
          "Second, real-world applications have proven its effectiveness.",
          "Finally, the counterarguments lack substantial empirical backing."
        ],
        saul: [
          "While my opponent makes compelling points, we must consider alternative perspectives.",
          "The studies referenced are limited in scope and don't account for diverse contexts.",
          "Practical application has shown mixed results across different demographics.",
          "Additionally, several experts in the field have raised legitimate concerns."
        ]
      });
      
      setIsAnalyzing(false);
      setShowResults(true);
    }, 3000);
  };
  
  const resetForm = () => {
    setShowResults(false);
    setDebateInput('');
    setDebateResults({
      harvey: [],
      saul: []
    });
  };

  // Example topics for quick selection
  const exampleTopics = [
    "Is social media beneficial for society?",
    "Should artificial intelligence be regulated?",
    "Is remote work better than working in an office?",
    "Should college education be free?"
  ];

  const selectExample = (example) => {
    setDebateInput(example);
  };

  return (
    <div className="pt-24 pb-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        {/* Header Section */}
        <div className="flex items-center gap-3 mb-8">
          <Link to="/" className="text-gray-500 hover:text-purple-600 transition-colors">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-3xl font-bold text-gray-800">The Courtroom</h1>
          <span className="bg-purple-100 text-purple-800 text-xs font-medium px-2.5 py-0.5 rounded">Debate Arena</span>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-10">
          {/* Left Column - Sentinel Info */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-6 sticky top-24">
              <div className="w-20 h-20 mx-auto bg-gradient-to-b from-purple-600 to-purple-800 rounded-full flex items-center justify-center mb-4 shadow-lg">
                <Scale size={32} className="text-white" />
              </div>
              
              <h2 className="text-xl font-bold text-center text-gray-800 mb-2">The Courtroom</h2>
              <p className="text-center text-purple-600 font-medium text-sm mb-4">Perspectives Debate</p>
              
              <div className="flex justify-between items-center mb-6">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-2">
                    <img src="https://via.placeholder.com/50" alt="Harvey Specter" className="rounded-full" />
                  </div>
                  <p className="text-xs font-medium text-gray-800">Harvey Specter</p>
                </div>
                
                <div className="flex-shrink-0 text-gray-400">VS</div>
                
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto bg-red-100 rounded-full flex items-center justify-center mb-2">
                    <img src="https://via.placeholder.com/50" alt="Saul Goodman" className="rounded-full" />
                  </div>
                  <p className="text-xs font-medium text-gray-800">Saul Goodman</p>
                </div>
              </div>
              
              <div className="border-t border-gray-100 pt-4 mt-2">
                <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                  <Gavel size={16} />
                  <span>How The Courtroom Helps You</span>
                </h3>
                <ul className="text-gray-600 text-sm space-y-2">
                  <li className="flex items-start gap-2">
                    <Lightbulb size={16} className="text-yellow-500 mt-0.5 shrink-0" />
                    <span>See multiple perspectives on complex issues</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Lightbulb size={16} className="text-yellow-500 mt-0.5 shrink-0" />
                    <span>Understand nuanced arguments from both sides</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Lightbulb size={16} className="text-yellow-500 mt-0.5 shrink-0" />
                    <span>Explore the grey areas between opposing viewpoints</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Lightbulb size={16} className="text-yellow-500 mt-0.5 shrink-0" />
                    <span>Make more informed decisions with balanced information</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
          
          {/* Right Column - Main Content */}
          <div className="lg:col-span-2">
            {!isAnalyzing && !showResults ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-purple-600 to-purple-800 p-6 text-white">
                  <h2 className="text-xl font-bold mb-2">Courtroom Debate</h2>
                  <p className="opacity-90">Enter a topic or statement to hear arguments from both sides</p>
                </div>
                
                <div className="p-6">
                  <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                      <label htmlFor="debate-input" className="block text-sm font-medium text-gray-700 mb-1">
                        Topic or Statement
                      </label>
                      <textarea
                        id="debate-input"
                        className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-300 focus:border-purple-500 transition-all duration-300 resize-none"
                        placeholder="Enter a topic, question, or statement you want analyzed from multiple perspectives..."
                        rows={4}
                        value={debateInput}
                        onChange={(e) => setDebateInput(e.target.value)}
                      ></textarea>
                    </div>
                    
                    <button
                      type="submit"
                      className="bg-gradient-to-r from-purple-600 to-purple-800 text-white font-medium py-2 px-4 rounded-md hover:from-purple-700 hover:to-purple-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 w-full flex items-center justify-center gap-2"
                    >
                      <Gavel size={18} />
                      Start Debate
                    </button>
                  </form>
                  
                  <div className="mt-6">
                    <h3 className="text-sm font-semibold text-gray-700 mb-3">Try these examples:</h3>
                    <div className="flex flex-wrap gap-2">
                      {exampleTopics.map((example, index) => (
                        <button
                          key={index}
                          onClick={() => selectExample(example)}
                          className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-800 py-1.5 px-3 rounded-full transition-colors"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-6 bg-purple-50 border border-purple-100 rounded-md p-4">
                    <h3 className="text-sm font-semibold text-purple-800 mb-2 flex items-center gap-2">
                      <AlertTriangle size={16} />
                      <span>How to Use The Courtroom</span>
                    </h3>
                    <ol className="text-sm text-purple-700 space-y-1 list-decimal pl-5">
                      <li>Enter a topic, question, or statement you want to explore</li>
                      <li>Click "Start Debate" to begin the analysis</li>
                      <li>Harvey Specter will present arguments from one perspective</li>
                      <li>Saul Goodman will counter with alternative viewpoints</li>
                      <li>Consider both sides to form your own nuanced opinion</li>
                    </ol>
                  </div>
                </div>
              </div>
            ) : isAnalyzing ? (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 p-8 flex flex-col items-center justify-center min-h-[400px]">
                <div className="animate-pulse">
                  <div className="w-20 h-20 bg-gradient-to-b from-purple-600 to-purple-800 rounded-full flex items-center justify-center shadow-lg">
                    <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                  </div>
                </div>
                
                <h3 className="mt-6 text-xl font-bold text-gray-800">Preparing Arguments</h3>
                <p className="mt-2 text-gray-600 max-w-md text-center">
                  Our attorneys are preparing their arguments for this topic. This may take a moment...
                </p>
                
                <div className="mt-8 w-full max-w-md bg-gray-200 rounded-full h-2.5">
                  <div className="bg-purple-600 h-2.5 rounded-full animate-pulse w-[60%]"></div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
                <div className="bg-gradient-to-r from-purple-600 to-purple-800 p-6 text-white flex justify-between items-center">
                  <div>
                    <h2 className="text-xl font-bold mb-1">Debate Results</h2>
                    <p className="opacity-90 text-sm">Multiple perspectives on your topic</p>
                  </div>
                  <button
                    onClick={resetForm}
                    className="bg-white/20 hover:bg-white/30 text-white font-medium py-1.5 px-3 rounded-md transition-colors text-sm flex items-center gap-1.5"
                  >
                    <ArrowLeft size={16} />
                    New Debate
                  </button>
                </div>
                
                <div className="p-6">
                  <div className="mb-6 border border-gray-200 rounded-md p-4 bg-gray-50">
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Topic</h3>
                    <p className="text-gray-700">{debateInput}</p>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {/* Harvey Specter Arguments */}
                    <div className="border rounded-md overflow-hidden">
                      <div className="bg-blue-600 p-3 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center flex-shrink-0">
                          <img src="https://via.placeholder.com/40" alt="Harvey Specter" className="rounded-full" />
                        </div>
                        <div>
                          <h3 className="text-white font-semibold">Harvey Specter</h3>
                          <p className="text-blue-100 text-xs">Corporate Attorney</p>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-blue-50">
                        <h4 className="font-medium text-blue-900 mb-3">Key Arguments</h4>
                        <div className="space-y-3">
                          {debateResults.harvey.map((point, index) => (
                            <div key={index} className="flex items-start gap-2">
                              <div className="bg-blue-100 text-blue-600 rounded-full w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                                {index + 1}
                              </div>
                              <p className="text-gray-700 text-sm">{point}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    
                    {/* Saul Goodman Arguments */}
                    <div className="border rounded-md overflow-hidden">
                      <div className="bg-red-600 p-3 flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center flex-shrink-0">
                          <img src="https://via.placeholder.com/40" alt="Saul Goodman" className="rounded-full" />
                        </div>
                        <div>
                          <h3 className="text-white font-semibold">Saul Goodman</h3>
                          <p className="text-red-100 text-xs">Criminal Defense Lawyer</p>
                        </div>
                      </div>
                      
                      <div className="p-4 bg-red-50">
                        <h4 className="font-medium text-red-900 mb-3">Key Arguments</h4>
                        <div className="space-y-3">
                          {debateResults.saul.map((point, index) => (
                            <div key={index} className="flex items-start gap-2">
                              <div className="bg-red-100 text-red-600 rounded-full w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                                {index + 1}
                              </div>
                              <p className="text-gray-700 text-sm">{point}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mb-6 border-t border-b border-gray-100 py-4">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                      <Scale size={18} className="text-purple-600" />
                      <span>Balanced Perspective</span>
                    </h3>
                    <div className="bg-purple-50 border border-purple-100 rounded-md p-4">
                      <p className="text-gray-700">
                        This topic has valid perspectives on both sides. Harvey emphasizes evidence-based reasoning and established precedents, 
                        while Saul highlights contextual factors and alternative interpretations. The truth likely contains elements from both viewpoints, 
                        and the most balanced approach would consider the strengths of each argument.
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">Key Considerations</h3>
                    <div className="space-y-3">
                      <ConsiderationItem 
                        title="Consider the source of information"
                        description="Both sides may cite sources that favor their position. Evaluate the credibility of sources used in each argument."
                      />
                      
                      <ConsiderationItem 
                        title="Look for common ground"
                        description="Despite their differences, both perspectives often share some fundamental agreements that can serve as a starting point."
                      />
                      
                      <ConsiderationItem 
                        title="Context matters"
                        description="What's true in one context may not apply universally. Consider how different circumstances might affect each argument."
                      />
                      
                      <ConsiderationItem 
                        title="Watch for logical fallacies"
                        description="Both sides may employ rhetorical techniques that sound convincing but don't actually support their conclusion."
                      />
                    </div>
                  </div>
                  
                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <button className="bg-gradient-to-r from-blue-600 to-blue-800 text-white font-medium py-2 px-4 rounded-md hover:from-blue-700 hover:to-blue-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center justify-center gap-2">
                      <span>Download Harvey's Brief</span>
                    </button>
                    
                    <button className="bg-gradient-to-r from-red-600 to-red-800 text-white font-medium py-2 px-4 rounded-md hover:from-red-700 hover:to-red-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center justify-center gap-2">
                      <span>Download Saul's Brief</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ConsiderationItem = ({ title, description }) => {
  return (
    <div className="flex items-start gap-3">
      <div className="bg-purple-100 p-2 rounded-full flex-shrink-0 mt-0.5">
        <Lightbulb size={16} className="text-purple-700" />
      </div>
      <div>
        <h4 className="font-medium text-gray-800">{title}</h4>
        <p className="text-sm text-gray-600 mt-0.5">{description}</p>
      </div>
    </div>
  );
};

export default CourtroomPage;