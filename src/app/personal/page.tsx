import PersonalHero from "./components/PersonalHero";
import SoccerWriteupSection from "./components/SoccerWriteupSection";
import RunningWriteupSection from "./components/RunningWriteupSection";
import SoundersMediaSection from "./components/SoundersMediaSection";
import RunningMediaSection from "./components/RunningMediaSection";
import SoccerMediaSection from "./components/SoccerMediaSection";

export default function PersonalPage() {
  return (
    <main className="min-h-screen">
      <PersonalHero />
      <SoccerWriteupSection />
      <RunningWriteupSection />
      <SoundersMediaSection />
      <RunningMediaSection />
      <SoccerMediaSection />
    </main>
  );
}