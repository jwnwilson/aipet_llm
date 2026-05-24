import { Button } from './ui/button'

export function AccessPending() {
  return (
    <div className="flex flex-col items-center justify-center h-screen gap-5 px-6 text-center">
      <div className="font-['IBM_Plex_Mono'] text-[0.65rem] uppercase tracking-[0.18em] text-[#888888]">
        Access pending
      </div>
      <h1 className="font-['DM_Serif_Display'] text-[2.5rem] leading-tight text-[#1a1a1a] max-w-md">
        Awaiting approval
      </h1>
      <p className="font-['Outfit'] text-[0.95rem] text-[#3a3a36] max-w-md leading-relaxed">
        Your account has not been approved yet. Contact an administrator to gain access
        to the training dashboard.
      </p>
      <div className="pt-3">
        <Button variant="outline" onClick={() => window.location.reload()}>
          Refresh
        </Button>
      </div>
    </div>
  )
}
