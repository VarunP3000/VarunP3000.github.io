import PersonalHero from "./components/PersonalHero";

import SoccerWriteupSection from "./components/SoccerWriteupSection";
import RunningWriteupSection from "./components/RunningWriteupSection";

import RunningMediaSection from "./components/RunningMediaSection";
import SoccerMediaSection from "./components/SoccerMediaSection";

export default function PersonalPage() {
  return (
    <main className="min-h-screen text-white [&_*]:!text-white">
      <PersonalHero />

      <SoccerWriteupSection />

      <RunningWriteupSection />

      <RunningMediaSection />

      <SoccerMediaSection />
    </main>
  );
}