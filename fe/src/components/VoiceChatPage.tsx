
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Phone, PhoneOff, Mic, MicOff } from "lucide-react";
import { Link } from "react-router-dom";
import CallAnimation from "./CallAnimation";

const VoiceChatPage = () => {
  const [selectedScenario, setSelectedScenario] = useState("");
  const [selectedVoice, setSelectedVoice] = useState("");
  const [isCallActive, setIsCallActive] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [emotion, setEmotion] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

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

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTranscript(data.text);
      setEmotion(data.emotion);
    };

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0 && ws.readyState === 1) {
        e.data.arrayBuffer().then(buffer => ws.send(buffer));
      }
    };

    mediaRecorder.start(250);
    setIsCallActive(true);
  };

  const handleEndCall = () => {
  setIsCallActive(false);
  setIsMuted(false);

  // Stop recording
  mediaRecorderRef.current?.stop();

  // Stop microphone stream
  if (mediaRecorderRef.current?.stream) {
    mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
  }

  // Close WebSocket
  wsRef.current?.close();

  // Clear refs
  mediaRecorderRef.current = null;
  wsRef.current = null;
};


  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      <nav className="p-6 flex justify-between items-center">
        <Link to="/">
          <h1 className="text-2xl font-bold text-white hover:text-cyan-400 transition-colors">VoiceAI</h1>
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
              <h1 className="text-4xl font-bold text-white mb-4">
                Start Your AI Conversation
              </h1>
              <p className="text-xl text-gray-300">
                Choose your scenario and voice preference to begin
              </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <Card className="bg-white/10 backdrop-blur-lg border-white/20">
                <CardHeader>
                  <CardTitle className="text-white text-2xl">Select Scenario</CardTitle>
                  <CardDescription className="text-gray-300">
                    Choose the type of conversation you'd like to have
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={selectedScenario} onValueChange={setSelectedScenario}>
                    <SelectTrigger className="bg-white/10 border-white/30 text-white">
                      <SelectValue placeholder="Choose a scenario..." />
                    </SelectTrigger>
                    <SelectContent>
                      {scenarios.map((scenario) => (
                        <SelectItem key={scenario.value} value={scenario.value}>
                          {scenario.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>

              <Card className="bg-white/10 backdrop-blur-lg border-white/20">
                <CardHeader>
                  <CardTitle className="text-white text-2xl">Select Voice</CardTitle>
                  <CardDescription className="text-gray-300">
                    Choose your preferred AI voice personality
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={selectedVoice} onValueChange={setSelectedVoice}>
                    <SelectTrigger className="bg-white/10 border-white/30 text-white">
                      <SelectValue placeholder="Choose a voice..." />
                    </SelectTrigger>
                    <SelectContent>
                      {voices.map((voice) => (
                        <SelectItem key={voice.value} value={voice.value}>
                          {voice.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>
            </div>

            <div className="text-center">
              <Button
                size="lg"
                onClick={handleStartCall}
                disabled={!selectedScenario || !selectedVoice}
                className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-12 py-6 text-xl rounded-full transition-all duration-300 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
              >
                <Phone className="w-6 h-6 mr-3" />
                Start Conversation
              </Button>
              {(!selectedScenario || !selectedVoice) && (
                <p className="text-yellow-300 mt-4">
                  Please select both a scenario and voice to continue
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold text-white mb-2">Call in Progress</h1>
              <p className="text-gray-300">
                {scenarios.find(s => s.value === selectedScenario)?.label} • {voices.find(v => v.value === selectedVoice)?.label}
              </p>
            </div>

            <CallAnimation />

            <div className="text-center mb-8">
              <div className="inline-flex items-center bg-green-500/20 text-green-300 px-6 py-3 rounded-full">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse mr-3"></div>
                Connected and Ready
              </div>
            </div>

            <div className="flex justify-center space-x-6">
              <Button
                variant="outline"
                size="lg"
                onClick={toggleMute}
                className={`${isMuted 
                  ? 'bg-red-500/20 border-red-500 text-red-300 hover:bg-red-500/30' 
                  : 'bg-white/10 border-white/30 text-white hover:bg-white/20'} px-8 py-4 rounded-full transition-all duration-300`}
              >
                {isMuted ? <MicOff className="w-6 h-6 mr-2" /> : <Mic className="w-6 h-6 mr-2" />}
                {isMuted ? 'Unmute' : 'Mute'}
              </Button>
              <Button
                size="lg"
                onClick={handleEndCall}
                className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white px-8 py-4 rounded-full transition-all duration-300 hover:scale-105"
              >
                <PhoneOff className="w-6 h-6 mr-2" />
                End Call
              </Button>
            </div>

            <div className="mt-12 text-center">
              <Card className="bg-white/10 backdrop-blur-lg border-white/20 max-w-2xl mx-auto">
                <CardContent className="p-6">
                  <p className="text-gray-300 text-lg">
                    You're now connected! Start speaking and the AI will respond naturally.
                    Use the mute button if you need a moment, or end the call when you're finished.
                  </p>
                </CardContent>
              </Card>
            </div>

            <div className="text-center mt-8">
              <h2 className="text-xl text-white font-bold">Live Transcript:</h2>
              <p className="text-lg text-white mt-2">{transcript || "---"}</p>
              <h2 className="text-xl text-white font-bold mt-6">Detected Emotion:</h2>
              <p className="text-lg text-pink-300 mt-2">{emotion || "---"}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceChatPage;
