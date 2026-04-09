export default function SoccerWriteupSection() {
    return (
      <section className="w-full px-6 py-16 md:px-12 lg:px-24">
        <div className="mx-auto max-w-6xl">
          <div className="grid gap-8 md:grid-cols-12">
            <div className="md:col-span-3">
              <h2 className="text-2xl font-semibold md:text-3xl">Soccer</h2>
            </div>
  
            <div className="md:col-span-9">
              <div className="rounded-2xl border border-gray-200 p-6 md:p-8">
                <p className="text-base leading-8 text-gray-700">
                I have played competitive soccer since I was 8 years old, including time with ISC Gunners and Eastside FC. 
                Through this experience, I developed strong collaboration skills and the ability to make decisions under pressure. 
                One of the most meaningful moments in my experience was winning the Washington State Cup, which reflected a culmination of years of effort and discipline.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    );
  }