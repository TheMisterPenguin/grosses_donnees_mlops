"use client"
import { QueryClientProvider as QCP, QueryClient } from "@tanstack/react-query";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
const queryClient = new QueryClient();

export function QueryClientProvider({children} : Readonly<{children: React.ReactNode}>) {
	return (
		// Provide the client to your App
		<QCP client={queryClient}>
			<ReactQueryDevtools initialIsOpen={false} />
			{children}
		</QCP>
	);
}