export default function SoccerMediaSection() {
    return (
      <section className="w-full px-6 py-16 md:px-12 lg:px-24">
        <div className="mx-auto max-w-6xl space-y-10">
          
          {/* Header */}
          <div className="space-y-3">
            <p className="text-sm uppercase tracking-[0.2em] text-gray-500">
              Experience
            </p>
            <h2 className="text-2xl font-semibold md:text-3xl">
              Soccer Experience
            </h2>
            <p className="max-w-3xl text-base leading-7 text-gray-600">
              Highlights from my competitive soccer experience, including key in-game moments and achievements.
            </p>
          </div>
  
          {/* Supporting Image */}
          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="overflow-hidden rounded-2xl border border-gray-200">
              <img
                src="/images/soccer_photo.jpg"
                alt="Holding the state cup"
                className="h-full w-full object-cover"
              />
            </div>
  
            <div className="text-gray-600 text-base leading-7">
              <p>
                One of the most meaningful moments from my soccer experience was winning the state cup, representing the culmination of years of training, teamwork, and competition.
              </p>
            </div>
          </div>

          {/* Highlight Reel (MAIN FOCUS) */}
          <div className="overflow-hidden rounded-2xl border border-gray-200">
            <video controls className="w-full">
              <source src="/videos/SoccerHighlights.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
  
        </div>
      </section>
    );
  }