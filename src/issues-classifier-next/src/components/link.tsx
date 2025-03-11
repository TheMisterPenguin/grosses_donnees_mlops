"use client"
import NextLink from "next/link";
import { usePathname } from "next/navigation";
import { NavigationMenu } from "radix-ui";
import type { FC } from 'react';

export const Link: FC<Parameters<typeof NavigationMenu.Link>[0] & Parameters<typeof NextLink>[0]> = ({href, ...props}) => {
	const pathname = usePathname();
	const isActive = href === pathname;

	return (
		<NavigationMenu.Link asChild active={isActive}>
			<NextLink href={href} className="NavigationMenuLink" {...props} />
		</NavigationMenu.Link>
	);
};
