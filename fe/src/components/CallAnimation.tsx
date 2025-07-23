
import { useEffect, useState } from "react";

const CallAnimation = () => {
  const [audioLevels, setAudioLevels] = useState<number[]>([]);

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate audio levels with random values
      const newLevels = Array.from({ length: 20 }, () => Math.random() * 100);
      setAudioLevels(newLevels);
    }, 150);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-16">
      {/* Main Audio Visualizer */}
      <div className="relative mb-12">
        {/* Outer Ring */}
        <div className="w-80 h-80 rounded-full border-4 border-cyan-400/30 animate-pulse"></div>
        
        {/* Middle Ring */}
        <div className="absolute top-6 left-6 w-68 h-68 rounded-full border-4 border-purple-400/40 animate-pulse" style={{ animationDelay: '0.5s' }}></div>
        
        {/* Inner Circle */}
        <div className="absolute top-12 left-12 w-56 h-56 rounded-full bg-gradient-to-r from-cyan-400/20 to-purple-400/20 backdrop-blur-lg flex items-center justify-center">
          <div className="w-32 h-32 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 flex items-center justify-center animate-pulse">
            <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 animate-bounce"></div>
            </div>
          </div>
        </div>

        {/* Floating Particles */}
        {[...Array(8)].map((_, index) => (
          <div
            key={index}
            className={`absolute w-4 h-4 bg-cyan-400 rounded-full animate-ping`}
            style={{
              top: `${20 + Math.sin(index * 45 * Math.PI / 180) * 140}px`,
              left: `${20 + Math.cos(index * 45 * Math.PI / 180) * 140}px`,
              animationDelay: `${index * 0.2}s`,
              animationDuration: '2s'
            }}
          ></div>
        ))}
      </div>

      {/* Audio Bars Visualizer */}
      <div className="flex items-end justify-center space-x-2 h-24 mb-8">
        {audioLevels.map((level, index) => (
          <div
            key={index}
            className="bg-gradient-to-t from-cyan-500 to-purple-500 rounded-t-lg transition-all duration-150 ease-out"
            style={{
              width: '8px',
              height: `${Math.max(level * 0.8, 10)}px`,
              opacity: 0.7 + (level / 100) * 0.3
            }}
          ></div>
        ))}
      </div>

      {/* Status Text */}
      <div className="text-center">
        <div className="flex items-center justify-center mb-4">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
        <p className="text-white text-xl font-medium mb-2">AI is listening...</p>
        <p className="text-gray-300">Speak naturally and wait for the response</p>
      </div>

      {/* Waveform Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400/20 to-transparent animate-pulse"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-full h-1 bg-gradient-to-r from-transparent via-purple-400/20 to-transparent animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>
    </div>
  );
};

export default CallAnimation;
