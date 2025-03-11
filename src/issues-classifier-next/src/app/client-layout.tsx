"use client";
import { Link } from "@/components/link";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { NavigationMenu, NavigationMenuItem, NavigationMenuList, navigationMenuTriggerStyle } from "@/components/ui/navigation-menu";

export function ClientLayout({ children }: Readonly<{ children: React.ReactNode }>) {
    return <div className="">
                <header className="border-grid border-dashed sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                    <div className="ml-auto mr-auto w-full max-w-[1400px] border-dashed" style={{borderColor: "hsl(var(--border))"}}>
                        <div className="container flex h-14 items-center gap-2 md:gap-4">
                            <NavigationMenu>
                                <NavigationMenuList>
                                    <NavigationMenuItem>
                                        <Link href="/" className={navigationMenuTriggerStyle()}>
                                            Accueil
                                        </Link>
                                        <Link href="/#about" className={navigationMenuTriggerStyle()}>
                                            À propos
                                        </Link>
                                    </NavigationMenuItem>
                                </NavigationMenuList>
                            </NavigationMenu>
                        </div>
                    </div>
                </header>
                <main className="container min-w-[100vw]">
                    <AuroraBackground>
                        {children}
                    </AuroraBackground>
                    {/* </div> */}
                </main>
            </div>
        ;
}