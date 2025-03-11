"use client";
import { Button } from "@/components/ui/button";
import ColourfulText from "@/components/ui/colourful-text";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";

export default function Home() {
	return (
		<>
			<div className="relative">
				<h1 className="relative text-8xl font-bold text-center">
					Attribution <ColourfulText text="automatique" /> <br /> de labels pour issues GitHub
				</h1>
			</div>
			<div className="relative container p-12 flex justify-center w-full">
				<IssueForm />
			</div>
		</>
	);
}

function IssueForm() {
	const router = useRouter();

	const action = (data: FormData) => {
		console.debug(data.get("url"));

		if (!data.has("url")) return;

		const regex = /https:\/\/github\.com\/(.+)\/(.+)\/issues\/([0-9]+)/;

		const url = data.get("url") as string;


		const m = regex.exec(url);
		if (m === null) {
			console.error("Invalid URL");
			return;
		}

		const owner = m[1] as string;
		const repo = m[2] as string;
		const issue = m[3] as string;

		console.log(owner, repo, issue);

		const params = new URLSearchParams({owner, repo, issue});

		router.push(`issue/?${params.toString()}`);
	};

	return (
		<form action={action}>
			<div className="flex w-fit items-center justify-center h-[4.5rem] px-0 py-0 bg-transparent">
				<Input name="url" className="h-full rounded-r-none w-[48rem] rounded-l-full" type="text" placeholder="URL d'une issue" />
				<Button className="h-full rounded-l-none rounded-r-full w-48 px-0 py-0" type="submit">
					C'est parti !
				</Button>
			</div>
		</form>
	);
}
