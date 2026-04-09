export default function SoundersMediaSection() {
    return (
      <section className="w-full px-6 py-16 md:px-12 lg:px-24">
        <div className="mx-auto max-w-6xl space-y-8">
          <div className="space-y-3">
            <p className="text-sm uppercase tracking-[0.2em] text-gray-500">
              Media
            </p>
            <h2 className="text-2xl font-semibold md:text-3xl">
              Love for the Sounders
            </h2>
            <p className="max-w-3xl text-base leading-7 text-gray-600">
              Use this section for Sounders-related pictures or videos.
            </p>
          </div>
  
          <div className="grid gap-6 md:grid-cols-2">
            <div className="overflow-hidden rounded-2xl border border-gray-200">
              <img
                src="/images/Sounders_pic.jpg"
                alt="Sounders related moment"
                className="h-full w-full object-cover"
              />
            </div>
            <div className="overflow-hidden rounded-2xl border border-gray-200">
              <img
                src="/images/CWC_pic.jpg"
                alt="Sounders related moment"
                className="h-full w-full object-cover"
              />
            </div>
            <div className="overflow-hidden rounded-2xl border border-gray-200">
              <video controls className="h-full w-full">
                <source src="/videos/Sounders_chants.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="overflow-hidden rounded-2xl border border-gray-200">
              <video controls className="h-full w-full">
                <source src="/videos/LeagueCupCelebrations.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      </section>
    );
  }