import { NextRequest, NextResponse } from 'next/server';
import { Octokit } from "octokit";
import { z } from "zod";
const paramsSchema = z.object({
    owner : z.string(),
    repo : z.string(),
    issueNumber : z.coerce.number()
})

const resSchema = z.object({
	predicted_labels: z.array(z.string())
});

export async function GET(request: NextRequest) {
    const params = Object.fromEntries(request.nextUrl.searchParams.entries());
    
    let parsedParams 
    try {
        parsedParams = await paramsSchema.parseAsync(params);
    }
    catch (e) {

        console.error("Error while parsing params", e)
        return NextResponse.json({"message" : "Invalid params"}, {status : 400})
    }

    // On récupère les infos depuis GitHub
    const octokit = new Octokit();

    let gitRes;
    try {
		gitRes = await octokit.rest.issues.get({owner: parsedParams.owner, repo: parsedParams.repo, issue_number: parsedParams.issueNumber});
	} catch {
		return NextResponse.json({message: "Repo not found"}, {status: 404});
	}

    const body : string | null = gitRes.data.body_text ?? gitRes.data.body ?? gitRes.data.body_html ?? null;

    if(body === null){
        return NextResponse.json({message: "Issue body empty"}, {status: 400});
    }
    
    console.debug("Res body : ",body);

    // On appelle le modèle

    const res = await fetch(`${process.env["API_URL"]}/predict`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			text: body,
		}),
	});

    const parsedRes = await resSchema.parseAsync(await res.json())

    console.debug(parsedRes);

    return Response.json(parsedRes.predicted_labels);
}
