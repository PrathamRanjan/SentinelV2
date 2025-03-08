import { Link } from 'react-router-dom';
import { Shield, Search, FileText, Gavel, Headphones } from 'lucide-react';

const HomePage = () => {
  return (
    <div className="pt-24 pb-12">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        {/* Hero Section */}
        <div className="relative overflow-hidden bg-gradient-to-r from-cyan-700 to-cyan-900 rounded-2xl p-8 lg:p-12 mb-16 shadow-xl">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2 blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-64 h-64 bg-orange-500/10 rounded-full translate-y-1/2 -translate-x-1/2 blur-3xl"></div>
          
          <div className="relative z-10 max-w-3xl">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Choose Your <span className="text-orange-300">Sentinel</span>
            </h1>
            <p className="text-cyan-100 text-lg md:text-xl mb-8">
              Our team of legendary detectives is here to help you uncover the truth.
              Each sentinel has their own special abilities to assist you.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link to="/batman" className="bg-gradient-to-r from-cyan-600 to-cyan-800 text-white font-medium py-2 px-4 rounded-md hover:from-cyan-700 hover:to-cyan-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center gap-2">
                <Shield size={18} />
                <span>Meet Our Sentinels</span>
              </Link>
              <Link to="/courtroom" className="bg-white/20 hover:bg-white/30 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 flex items-center gap-2">
                <Gavel size={18} />
                <span>Enter The Courtroom</span>
              </Link>
            </div>
          </div>
        </div>

        {/* Sentinels Section */}
        <div className="mb-16">
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-gray-800">Choose Your Sentinel</h2>
            <div className="h-0.5 flex-grow ml-6 bg-gradient-to-r from-cyan-200 to-transparent"></div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <SentinelCard 
              name="Batman"
              role="YouTube Fact Checker"
              description="Batman specializes in analyzing YouTube videos for misinformation. His detective skills help identify misleading content."
              icon={<Shield className="w-6 h-6 text-cyan-500" />}
              color="from-blue-500 to-blue-700"
              link="/batman"
            />
            
            <SentinelCard 
              name="Sherlock Holmes"
              role="Text Fact Checker"
              description="With his legendary deductive reasoning, Sherlock analyzes text and statements to separate fact from fiction."
              icon={<Search className="w-6 h-6 text-cyan-500" />}
              color="from-indigo-500 to-purple-700"
              link="/sherlock"
            />
            
            <SentinelCard 
              name="Hercule Poirot"
              role="Deepfake Detector"
              description="This meticulous Belgian detective uses his 'little grey cells' to spot manipulated images, videos, and even cloned audio."
              icon={
                <div className="relative">
                  <FileText className="w-6 h-6 text-cyan-500" />
                  <Headphones className="w-3 h-3 text-cyan-500 absolute -bottom-1 -right-1" />
                </div>
              }
              color="from-teal-500 to-teal-700"
              link="/detective"
            />
          </div>
        </div>
        
        {/* Courtroom Section */}
        <div>
          <div className="flex items-center justify-between mb-8">
            <h2 className="text-2xl font-bold text-gray-800">Special Feature</h2>
            <div className="h-0.5 flex-grow ml-6 bg-gradient-to-r from-cyan-200 to-transparent"></div>
          </div>
          
          <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100">
            <div className="relative h-64 bg-gradient-to-r from-purple-700 to-purple-900 overflow-hidden">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-full h-full absolute top-0 left-0 overflow-hidden">
                  <div className="absolute top-1/2 left-1/4 w-32 h-32 bg-white/10 rounded-full transform -translate-y-1/2 blur-2xl"></div>
                  <div className="absolute top-1/2 right-1/4 w-32 h-32 bg-orange-500/10 rounded-full transform -translate-y-1/2 blur-2xl"></div>
                </div>
                <div className="relative z-10 p-8 text-center">
                  <h3 className="text-3xl font-bold text-white mb-2">The Courtroom</h3>
                  <p className="text-purple-200 text-lg max-w-2xl mx-auto">
                    Witness a debate between legendary attorneys Harvey Specter and Saul Goodman as they analyze both sides of any argument.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="p-6 flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <div className="bg-purple-100 p-3 rounded-lg">
                  <Gavel className="w-6 h-6 text-purple-700" />
                </div>
                <div>
                  <h4 className="text-lg font-semibold text-gray-800">See The Grey Areas</h4>
                  <p className="text-gray-600">Understand complex issues from multiple perspectives</p>
                </div>
              </div>
              
              <Link to="/courtroom" className="bg-gradient-to-r from-purple-600 to-purple-800 text-white font-medium py-2 px-4 rounded-md hover:from-purple-700 hover:to-purple-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 flex items-center gap-2 whitespace-nowrap">
                <Gavel size={18} />
                <span>Enter The Courtroom</span>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const SentinelCard = ({ name, role, description, icon, color, link }) => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-100 h-full hover:-translate-y-1 transition-transform duration-300">
      <div className={`relative h-48 bg-gradient-to-r ${color} overflow-hidden`}>
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Placeholder for character silhouette */}
          <div className="w-32 h-32 bg-white/20 rounded-full flex items-center justify-center">
            {icon}
          </div>
        </div>
      </div>
      
      <div className="p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-1">{name}</h3>
        <p className="text-cyan-600 font-medium text-sm mb-3">{role}</p>
        <p className="text-gray-600 mb-5">{description}</p>
        
        <Link to={link} className="bg-gradient-to-r from-cyan-600 to-cyan-800 text-white font-medium py-2 px-4 rounded-md hover:from-cyan-700 hover:to-cyan-900 transition-all duration-300 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 w-full flex items-center justify-center gap-2">
          <span>Choose {name}</span>
        </Link>
      </div>
    </div>
  );
};

export default HomePage;