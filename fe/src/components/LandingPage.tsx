
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Mic, Brain, Users, Zap } from "lucide-react";
import { Link } from "react-router-dom";

const LandingPage = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Conversations",
      description: "Experience natural, human-like interactions with advanced AI technology"
    },
    {
      icon: Mic,
      title: "Voice Selection",
      description: "Choose from various voice options including male and female personas"
    },
    {
      icon: Users,
      title: "Multiple Scenarios",
      description: "Select from different conversation scenarios tailored to your needs"
    },
    {
      icon: Zap,
      title: "Real-time Interaction",
      description: "Engage in seamless, real-time voice conversations with visual feedback"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      {/* Navigation */}
      <nav className="p-6 flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Socially</h1>
        <Link to="/chat">
          <Button variant="outline" className="text-white border-white hover:bg-white hover:text-purple-900">
            Try Now
          </Button>
        </Link>
      </nav>

      {/* Hero Section */}
      <div className="container mx-auto px-6 py-20 text-center">
        <div className="animate-fade-in">
          <h1 className="text-6xl font-bold text-white mb-8 leading-tight">
            Talk to AI Like You Would
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
              Talk to a Human
            </span>
          </h1>
          
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
            Experience the future of AI interaction with natural voice conversations, 
            multiple scenarios, and real-time visual feedback. Choose your preferred voice 
            and dive into meaningful AI dialogues.
          </p>

          <div className="space-x-4">
            <Link to="/chat">
              <Button size="lg" className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white px-8 py-6 text-lg rounded-full transition-all duration-300 hover:scale-105">
                Start Conversation
              </Button>
            </Link>
            <Button 
              variant="outline" 
              size="lg" 
              className="text-white border-white hover:bg-white hover:text-purple-900 px-8 py-6 text-lg rounded-full transition-all duration-300"
            >
              Learn More
            </Button>
          </div>
        </div>

        {/* Floating Animation */}
        <div className="mt-20 relative">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-64 h-64 bg-gradient-to-r from-cyan-400/20 to-purple-400/20 rounded-full animate-pulse"></div>
          </div>
          <div className="relative z-10 bg-white/10 backdrop-blur-lg rounded-2xl p-8 max-w-md mx-auto">
            <Mic className="w-16 h-16 text-white mx-auto mb-4 animate-bounce" />
            <p className="text-white text-lg">Voice-Enabled AI Ready</p>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-black/20 backdrop-blur-lg py-20">
        <div className="container mx-auto px-6">
          <h2 className="text-4xl font-bold text-white text-center mb-16">
            Why Choose VoiceAI?
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="bg-white/10 backdrop-blur-lg border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105">
                <CardHeader className="text-center">
                  <feature.icon className="w-12 h-12 text-cyan-400 mx-auto mb-4" />
                  <CardTitle className="text-white">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-gray-300 text-center">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="container mx-auto px-6 py-20 text-center">
        <h2 className="text-4xl font-bold text-white mb-8">
          Ready to Experience the Future?
        </h2>
        <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
          Join thousands of users who are already having meaningful conversations with AI
        </p>
        <Link to="/chat">
          <Button size="lg" className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white px-12 py-6 text-xl rounded-full transition-all duration-300 hover:scale-105">
            Get Started Now
          </Button>
        </Link>
      </div>
    </div>
  );
};

export default LandingPage;
