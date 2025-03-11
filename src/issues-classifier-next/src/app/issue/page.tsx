"use client";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import ColourfulText from "@/components/ui/colourful-text";
import { Label } from "@/components/ui/label";
import { MultiStepLoader } from "@/components/ui/multi-step-loader";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

export default function Boundary(){

	return <Suspense fallback={<div>Loading...</div>}>
		<Home />
	</Suspense>
}

function Home() {
	const search = useSearchParams();

	const {owner, repo, issue} = useMemo(() => {
		const owner = search.get("owner") as string;
		const repo = search.get("repo") as string;
		const issue = search.get("issue") as string;

		return {owner, repo, issue};
	}, [search]);

	const {data, isLoading} = useQuery({
		queryKey: ["issue", owner, repo, issue],
		queryFn: async () => {
			const res = await fetch(`/api/issue/?${new URLSearchParams({owner, repo, issueNumber: issue})}`, {
				method: "GET",
			});
			const data: Array<string> = await res.json();

			return data;
		},
		throwOnError: true,
	});

	return (
		<>
			<div className="relative">
				<h1 className="relative text-8xl font-bold text-center">
					Attribution <ColourfulText text="automatique" /> <br /> de labels pour issues GitHub
				</h1>
			</div>
			{/* <div className="dark:bg-black bg-white  dark:bg-grid-white/[0.2] bg-grid-black/[0.2] w-full flex flex-col items-center justify-center p-4 h-[50rem]"> */}
			<div className="relative container p-12 flex justify-center w-full">
				<MultiStepLoader
					loading={isLoading}
					duration={2000}
					loop={false}
					loadingStates={[
						{
							text: "Récupération des informations de l'issue",
						},
						{
							text: "Analyse…",
						},
					]}
				/>
			</div>
			<div className="relative container p-12 flex justify-center w-full">
				<Card>
					<CardHeader>
						<CardTitle><strong>Issue</strong></CardTitle>
					</CardHeader>
					<CardContent>
						<Label>Owner</Label>
						<p>{owner}</p>
						<Label>Repo</Label>
						<p>{repo}</p>
						<Label>Issue</Label>
						<p>{issue}</p>
					</CardContent>
					<CardFooter className="flex flex-col items-start gap-4">
						<Label>Labels</Label>
						<div>
							{data &&
								data.map(label => (
									<Badge key={label} variant={"outline"}>
										{label}
									</Badge>
								))}
						</div>
					</CardFooter>
				</Card>
			</div>
		</>
	);
}
