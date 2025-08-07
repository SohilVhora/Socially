import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Phone, PhoneOff, Mic, MicOff } from "lucide-react";
import { Link } from "react-router-dom";
import CallAnimation from "./CallAnimation";

// NEW: Define a type for our chat messages for better code quality
type Message = {
  text: string;
  sender: 'user' | 'ai';
  emotion?: string; // Optional: store the user's emotion with their message
};

const VoiceChatPage = () => {
  const [selectedScenario, setSelectedScenario] = useState("");
  const [selectedVoice, setSelectedVoice] = useState("");
  const [isCallActive, setIsCallActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  
  // --- STATE CHANGE: Replaced transcript and emotion with a messages array ---
  const [messages, setMessages] = useState<Message[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null); // Ref for scrolling

  // --- Scroll to bottom of chat when new messages are added ---
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const scenarios = [
    { value: "customer-service", label: "Customer Service Representative" },
    { value: "life-coach", label: "Life Coach & Mentor" },
    { value: "language-tutor", label: "Language Learning Tutor" },
    { value: "technical-support", label: "Technical Support Specialist" },
    { value: "therapist", label: "Virtual Therapist" },
    { value: "business-advisor", label: "Business Advisor" },
    { value: "creative-writer", label: "Creative Writing Partner" },
    { value: "interview-prep", label: "Interview Preparation Coach" }
  ];

  const voices = [
    { value: "female-professional", label: "Female - Professional" },
    { value: "female-friendly", label: "Female - Friendly" },
    { value: "male-professional", label: "Male - Professional" },
    { value: "male-casual", label: "Male - Casual" },
    { value: "female-energetic", label: "Female - Energetic" },
    { value: "male-calm", label: "Male - Calm" }
  ];

  const handleStartCall = async () => {
    if (!selectedScenario || !selectedVoice) return;

    // Reset message history for new call
    setMessages([]);

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    // --- WEBSOCKET LOGIC CHANGE: Handle both user and AI messages ---
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.text && data.emotion) {
        const userMessage: Message = { text: data.text, sender: 'user', emotion: data.emotion };
        setMessages((prev) => [...prev, userMessage]);
      } else if (data.ai_response) {
        const aiMessage: Message = { text: data.ai_response, sender: 'ai' };
        setMessages((prev) => [...prev, aiMessage]);
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0 && ws.readyState === 1) {
        e.data.arrayBuffer().then(buffer => {
          if (!isMuted) {
            ws.send(buffer)
          }
        });
      }
    };
    
    mediaRecorder.start(250);
    setIsCallActive(true);
  };

  const handleEndCall = () => {
    setIsCallActive(false);
    setIsMuted(false);
    mediaRecorderRef.current?.stop();
    if (mediaRecorderRef.current?.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
    wsRef.current?.close();
    mediaRecorderRef.current = null;
    wsRef.current = null;
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900 text-white">
      <nav className="p-6 flex justify-between items-center">
        <Link to="/">
          <h1 className="text-2xl font-bold hover:text-cyan-400 transition-colors">VoiceAI</h1>
        </Link>
        <Link to="/">
          <Button variant="outline" className="text-white border-white hover:bg-white hover:text-purple-900">
            Back to Home
          </Button>
        </Link>
      </nav>

      <div className="container mx-auto px-6 py-8">
        {!isCallActive ? (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-12">
              <h1 className="text-4xl font-bold mb-4">Start Your AI Conversation</h1>
              <p className="text-xl text-gray-300">Choose your scenario and voice preference to begin</p>
            </div>
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {/* Scenario and Voice Selection Cards remain the same */}
              <Card className="bg-white/10 backdrop-blur-lg border-white/20">
                <CardHeader>
                  <CardTitle className="text-white text-2xl">Select Scenario</CardTitle>
                  <CardDescription className="text-gray-300">Choose the type of conversation you'd like to have</CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                    <SelectTrigger className="bg-white/10 border-white/30 text-white"><SelectValue placeholder="Choose a scenario..." /></SelectTrigger>
                    <SelectContent>{scenarios.map((s) => <SelectItem key={s.value} value={s.value}>{s.label}</SelectItem>)}</SelectContent>
                  </Select>
                </CardContent>
              </Card>
              <Card className="bg-white/10 backdrop-blur-lg border-white/20">
                <CardHeader>
                  <CardTitle className="text-white text-2xl">Select Voice</CardTitle>
                  <CardDescription className="text-gray-300">Choose your preferred AI voice personality</CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                    <SelectTrigger className="bg-white/10 border-white/30 text-white"><SelectValue placeholder="Choose a voice..." /></SelectTrigger>
                    <SelectContent>{voices.map((v) => <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>)}</SelectContent>
                  </Select>
                </CardContent>
              </Card>
            </div>
            <div className="text-center">
              <Button size="lg" onClick={handleStartCall} disabled={!selectedScenario || !selectedVoice} className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-12 py-6 text-xl rounded-full transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100">
                <Phone className="w-6 h-6 mr-3" />
                Start Conversation
              </Button>
              {(!selectedScenario || !selectedVoice) && (<p className="text-yellow-300 mt-4">Please select both a scenario and voice to continue</p>)}
            </div>
          </div>
        ) : (
          // --- UI CHANGE: In-call view now shows a chat history ---
          <div className="max-w-4xl mx-auto flex flex-col h-[calc(100vh-150px)]">
            <div className="text-center mb-4">
              <h1 className="text-3xl font-bold mb-2">Call in Progress</h1>
              <p className="text-gray-300">{scenarios.find(s => s.value === selectedScenario)?.label}</p>
            </div>

            {/* NEW: Chat history display */}
            <div ref={chatContainerRef} className="flex-grow p-4 mb-4 overflow-y-auto bg-black/20 rounded-lg shadow-inner space-y-4">
              {messages.map((msg, index) => (
                <div key={index} className={`flex items-end gap-2 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.sender === 'ai' && <div className="w-8 h-8 rounded-full bg-purple-500 flex-shrink-0"></div>}
                  <div className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-2 rounded-xl shadow ${msg.sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'}`}>
                    <p>{msg.text}</p>
                    {msg.sender === 'user' && msg.emotion && (
                      <p className="text-xs text-blue-200 mt-1 capitalize">{msg.emotion}</p>
                    )}
                  </div>
                </div>
              ))}
               {messages.length === 0 && (
                 <div className="text-center text-gray-400">
                   <p>You're connected! Start speaking to see the conversation here.</p>
                 </div>
               )}
            </div>

            {/* Call controls */}
            <div className="flex-shrink-0">
              <div className="flex justify-center space-x-6">
                <Button variant="outline" size="lg" onClick={toggleMute} className={`${isMuted ? 'bg-red-500/20 border-red-500 text-red-300 hover:bg-red-500/30' : 'bg-white/10 border-white/30 text-white hover:bg-white/20'} px-8 py-4 rounded-full transition-all duration-300`}>
                  {isMuted ? <MicOff className="w-6 h-6 mr-2" /> : <Mic className="w-6 h-6 mr-2" />}
                  {isMuted ? 'Unmute' : 'Mute'}
                </Button>
                <Button size="lg" onClick={handleEndCall} className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white px-8 py-4 rounded-full transition-all duration-300 hover:scale-105">
                  <PhoneOff className="w-6 h-6 mr-2" />
                  End Call
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceChatPage;
