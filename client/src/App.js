import { BrowserRouter, Routes, Route } from "react-router-dom";
import FileUpload from "./components/FileUpload";
import LandingPage from "./pages/LandingPage";
import HowItWorks from "./pages/HowItWorks";
import Features from "./pages/Features";
import AboutUs from "./pages/AboutUs";
import Home from "./pages/Home";
import Results from "./pages/Results";
import EvaluationReport from "./pages/EvaluationReport"; // import the report page


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/features" element={<Features />} />
        <Route path="/about" element={<AboutUs />} />
        <Route path="/home" element={<Home />} />
        <Route path="/upload" element={<FileUpload />} />
        <Route path="/results" element={<Results />} />
        <Route path="/evaluation-report" element={<EvaluationReport />} />{" "}
        {/* route to evaluation page */}
      </Routes>
    </BrowserRouter>
  );
}

export default App;
