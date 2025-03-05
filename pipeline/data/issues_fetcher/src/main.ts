import { Octokit } from "https://esm.sh/octokit?dts";
import { delay } from "jsr:@std/async";
import { load } from "jsr:@std/dotenv";
import { Kafka } from "https://esm.sh/kafkajs?dts";
await load({envPath: ".env.local", export: true}).then(loaded => {
	if (Object.keys(loaded).length === 0) throw Error(".env.local not found");
});
console.log(Deno.env.toObject());
const {repos, issues} = await import("./db.ts");

const octokit = new Octokit({auth: Deno.env.get("GH_TOKEN")});

if (!Deno.env.get("GH_TOKEN")) {
	throw new Error("Could not connect to GitHub : missing token");
}

octokit.log.info("Connected to GitHub");


async function getIssues(owner: string, repoName: string) {
	let page = 1;

	while (true) {
		const res = await octokit.rest.issues.listForRepo({state: "all", owner, repo: repoName, per_page: 100, page});

		// // On écrit dans mongoDB
		// issues.insertMany(res.data);


		// Envoi des données dans le topic "issues" de Kafka sur localhost:9092
		const kafka = new Kafka({
			clientId: 'issues-fetcher',
			brokers: ['localhost:9092']
		});

		const producer = kafka.producer();
		await producer.connect();

		for (const issue of res.data) {
			await producer.send({
				topic: 'issues',
				messages: [
					{ value: JSON.stringify(issue) }
				]
			});
		}

		await producer.disconnect();



		console.log(`\tAdded ${res.data.length} records`);

		// On regarde si on est rate limited
		if (res.headers["x-ratelimit-remaining"] === "0") {
			// On attend jusqu'à pouvoir reprendre
			const currentTime = Date.now();

			console.log(`\tBeing rate limited waiting : ${Intl.DateTimeFormat("FR-fr", {timeStyle: "long"}).format(Number(res.headers["x-ratelimit-reset"]))}`);
			await delay(Number(res.headers["x-ratelimit-reset"]) - currentTime + 5);
		}

		// On check si on a fini
		if (res.data.length < 100) {
			console.log("\tDone");
			break;
		}

		page += 1;
	}
}

for await (const doc of repos.find()) {
	console.log(`Fetching issues for repo ${doc.owner.login}/${doc.name} (${doc.id})`);
	await getIssues(doc.owner.login, doc.name);
}
