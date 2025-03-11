import { Kafka } from "https://esm.sh/kafkajs?dts";
import { Octokit } from "https://esm.sh/octokit?dts";
import { delay } from "jsr:@std/async";
import { load } from "jsr:@std/dotenv";
await load({envPath : ".env.local", export: true}).then(loaded => {
  if(Object.keys(loaded).length === 0) throw Error(".env.local not found")});
console.log(Deno.env.toObject())
const { repos } = await import("./db.ts");

const octokit = new Octokit({auth: Deno.env.get("GH_TOKEN")});

if (!Deno.env.get("GH_TOKEN")){
  throw new Error("Could not connect to GitHub : missing token")
}

octokit.log.info("Connected to GitHub");

// octokit.paginate(octokit.rest.search.repos, {q: "stars:>5000", sort: "updated", per_page: 100}).then();

function* accumulator() {
	let current = 10000;

	while (current > 1500) {
		yield [current, current - 250];
		current -= 250;
	}
}

async function getRepos(query : string) {

  let page = 1
  
  while (true) {
	const res = await octokit.rest.search.repos({q: query, sort: "updated", per_page: 100, page});

		// // On écrit dans mongoDB

		// repos.insertMany(res.data.items);


		// Envoi des données dans le topic "repo" de Kafka sur localhost:9092
		const kafka = new Kafka({
			clientId: 'repo-fetcher',
			brokers: ['localhost:9092']
		});

		const producer = kafka.producer();
		await producer.connect();

		await producer.send({
			topic: 'repo',
			messages: res.data.items.map(item => ({ value: JSON.stringify(item) }))
		});

		await producer.disconnect();




		console.log(`\tAdded ${res.data.items.length} records`);

		// On regarde si on est rate limited
		if (res.headers["x-ratelimit-remaining"] === "0") {
			// On attend jusqu'à pouvoir reprendre

			const currentTime = Date.now();

		console.log(`\tBeing rate limited waiting : ${Intl.DateTimeFormat("FR-fr", {timeStyle: "long"}).format(Number(res.headers["x-ratelimit-reset"]))}`);
		await delay(Number(res.headers["x-ratelimit-reset"]) - currentTime + 5);
	}

	// On check si on a fini
	if (res.data.items.length < 100) {
		console.log("\tDone");
		break;
	}

    if(page + 1 === 11){
        console.log("\tExiting because there are more than 1000 results");
				break;
    }

	page += 1;
  }
}

for (const [upper, lower] of accumulator()){
  console.log(`Fetching repos with stars between ${lower} and ${upper}`);

  await getRepos(`stars:${lower}..${upper}`);
}

// await getRepos(`stars:>10000`);